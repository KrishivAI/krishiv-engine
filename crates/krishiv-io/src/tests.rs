// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the request, statistics, options and schema-mapping types
//! (the format-level behaviour is covered by the conformance suite in
//! `krishiv-testkit`).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal128Type, Float64Type, Int16Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, Date32Array, Decimal128Array, Float32Array, Int8Array,
    Int32Array, LargeStringArray, ListArray, MapArray, RecordBatch, StringArray, StructArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit};
use futures::FutureExt;
use futures::future::BoxFuture;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::{OptionValue, SecretString};

use crate::*;

fn schema(fields: Vec<Field>) -> SchemaRef {
    Arc::new(Schema::new(fields))
}

fn f(name: &str, dt: DataType) -> Field {
    Field::new(name, dt, true)
}

fn col(n: &str) -> ColumnPath {
    ColumnPath::new(n)
}

// ---- requests --------------------------------------------------------------

#[test]
fn row_selections_merge_intersect_and_window() {
    let s = RowSelection::from_ranges([5..8, 0..2, 1..3, 9..9, 7..10]);
    assert_eq!(s.ranges(), &[0..3, 5..10]);
    assert_eq!(s.row_count(), 8);
    assert!(s.contains(2) && !s.contains(3) && s.contains(9) && !s.contains(10));
    assert_eq!(s.within(2..7), vec![0..1, 3..5]);
    let d = RowSelection::excluding(6, [1, 1, 4, 99]);
    assert_eq!(d.ranges(), &[0..1, 2..4, 5..6]);
    assert_eq!(s.intersect(&d).ranges(), &[0..1, 2..3, 5..6]);
    assert_eq!(RowSelection::excluding(3, [0, 1, 2]).row_count(), 0);
}

#[test]
fn negation_normal_form_pushes_not_to_leaves() {
    let e = PushdownExpr::Not(Box::new(PushdownExpr::And(vec![
        PushdownExpr::cmp(col("a"), CmpOp::Lt, ScalarValue::Int32(1)),
        PushdownExpr::Or(vec![
            PushdownExpr::IsNull(col("b")),
            PushdownExpr::In {
                column: col("c"),
                values: vec![ScalarValue::Int32(1), ScalarValue::Int32(2)],
                collation: Collation::Utf8Binary,
            },
        ]),
        PushdownExpr::Constant(true),
        PushdownExpr::StartsWith {
            column: col("s"),
            prefix: "x".into(),
            collation: Collation::Utf8Binary,
        },
    ])));
    let n = e.to_nnf();
    let PushdownExpr::Or(parts) = &n else {
        panic!("{n:?}")
    };
    assert_eq!(
        parts[0],
        PushdownExpr::cmp(col("a"), CmpOp::GtEq, ScalarValue::Int32(1))
    );
    assert!(
        matches!(&parts[1], PushdownExpr::And(x) if matches!(&x[0], PushdownExpr::IsNotNull(_)) && matches!(&x[1], PushdownExpr::And(_)))
    );
    assert_eq!(parts[2], PushdownExpr::Constant(false));
    assert!(matches!(parts[3], PushdownExpr::Not(_)));
    assert_eq!(e.columns().len(), 4);
    for op in [
        CmpOp::Eq,
        CmpOp::NotEq,
        CmpOp::Lt,
        CmpOp::LtEq,
        CmpOp::Gt,
        CmpOp::GtEq,
    ] {
        assert_eq!(op.negate().negate(), op);
    }
    let nn = PushdownExpr::Not(Box::new(PushdownExpr::Not(Box::new(
        PushdownExpr::IsNotNull(col("a")),
    ))));
    assert_eq!(nn.to_nnf(), PushdownExpr::IsNotNull(col("a")));
    assert_eq!(
        PushdownExpr::eq(col("a"), ScalarValue::Null).columns()[0].to_string(),
        "[a]"
    );
}

#[test]
fn projections_keep_nested_fields_in_schema_order() {
    let inner = Fields::from(vec![
        f("x", DataType::Int32),
        f("y", DataType::Int32),
        f("z", DataType::Int32),
    ]);
    let entries = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            f("value", DataType::Struct(inner.clone())),
        ])),
        false,
    );
    let s = schema(vec![
        f("a", DataType::Int32),
        f("st", DataType::Struct(inner.clone())),
        f(
            "l",
            DataType::List(Arc::new(f("item", DataType::Struct(inner.clone())))),
        ),
        f(
            "ll",
            DataType::LargeList(Arc::new(f("item", DataType::Struct(inner)))),
        ),
        f("m", DataType::Map(Arc::new(entries), false)),
    ]);
    let p = project_schema(
        &s,
        &[
            ColumnPath::nested(["st", "z"]),
            col("a"),
            ColumnPath::nested(["st", "x"]),
            ColumnPath::nested(["l", "y"]),
            ColumnPath::nested(["ll", "y"]),
            ColumnPath::nested(["m", "x"]),
        ],
    )
    .unwrap();
    let names: Vec<&str> = p.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["st", "a", "l", "ll", "m"]);
    let DataType::Struct(st) = p.field(0).data_type() else {
        panic!()
    };
    assert_eq!(
        st.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
        ["x", "z"]
    );
    assert!(project_schema(&s, &[col("nope")]).is_err());
    assert!(project_schema(&s, &[ColumnPath::nested(["a", "x"])]).is_err());
    assert!(project_schema(&s, &[ColumnPath::nested(["st", "nope"])]).is_err());
    let mut req = ScanRequest::new(Arc::clone(&s), vec![col("a")]);
    req.metadata_cols.row_index = true;
    let out = req.output_schema().unwrap();
    assert_eq!(out.field(1).name(), ROW_INDEX_COLUMN);
    assert_eq!(ScanRequest::all(s).projection.len(), 5);
    assert_eq!(ColumnPath::default().root(), "");
}

#[test]
fn batches_are_projected_like_their_schemas() {
    let inner = Fields::from(vec![f("x", DataType::Int32), f("y", DataType::Int32)]);
    let st = StructArray::new(
        inner.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(Int32Array::from(vec![3, 4])) as ArrayRef,
        ],
        None,
    );
    let item = f("item", DataType::Struct(inner.clone()));
    let list = ListArray::new(
        Arc::new(item.clone()),
        OffsetBuffer::from_lengths([1, 1]),
        Arc::new(st.clone()),
        None,
    );
    let large = arrow_array::LargeListArray::new(
        Arc::new(item),
        arrow_buffer::OffsetBuffer::<i64>::from_lengths([2, 0]),
        Arc::new(st.clone()),
        None,
    );
    let entries = Fields::from(vec![
        Field::new("key", DataType::Utf8, false),
        f("value", DataType::Struct(inner.clone())),
    ]);
    let kv = StructArray::new(
        entries.clone(),
        vec![
            Arc::new(StringArray::from(vec!["k1", "k2"])) as ArrayRef,
            Arc::new(st.clone()) as ArrayRef,
        ],
        None,
    );
    let entries_field = Arc::new(Field::new("entries", DataType::Struct(entries), false));
    let map = MapArray::new(
        Arc::clone(&entries_field),
        OffsetBuffer::from_lengths([1, 1]),
        kv,
        None,
        false,
    );
    let s = schema(vec![
        f("a", DataType::Int32),
        f("st", DataType::Struct(inner)),
        f("l", list.data_type().clone()),
        f("ll", large.data_type().clone()),
        f("m", DataType::Map(entries_field, false)),
    ]);
    let batch = RecordBatch::try_new(
        Arc::clone(&s),
        vec![
            Arc::new(Int32Array::from(vec![7, 8])),
            Arc::new(st),
            Arc::new(list),
            Arc::new(large),
            Arc::new(map),
        ],
    )
    .unwrap();
    let paths = [
        ColumnPath::nested(["m", "y"]),
        ColumnPath::nested(["st", "y"]),
        col("a"),
        ColumnPath::nested(["l", "x"]),
        ColumnPath::nested(["ll", "y"]),
    ];
    let out = project_batch(&batch, &paths).unwrap();
    assert_eq!(out.schema(), project_schema(&s, &paths).unwrap());
    assert_eq!(out.num_rows(), 2);
    let st = out.column(1).as_struct();
    assert_eq!(st.num_columns(), 1);
    assert_eq!(
        st.column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .values(),
        &[3, 4]
    );
    let l = out.column(3).as_list::<i32>();
    assert_eq!(l.values().as_struct().num_columns(), 1);
    // Whole columns are shared, not copied; a bad path is an error.
    let whole = project_batch(&batch, &[col("st")]).unwrap();
    assert!(Arc::ptr_eq(whole.column(0), batch.column(1)));
    assert!(project_batch(&batch, &[col("nope")]).is_err());
    let rows_only = project_batch(&batch, &[]).unwrap();
    assert_eq!((rows_only.num_columns(), rows_only.num_rows()), (0, 2));
    // List views and fixed-size lists of structs prune too.
    let item = Arc::new(f("item", DataType::Struct(inner2())));
    let st2 = StructArray::new(
        inner2(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            Arc::new(Int32Array::from(vec![5, 6, 7, 8])) as ArrayRef,
        ],
        None,
    );
    let view = arrow_array::ListViewArray::new(
        Arc::clone(&item),
        arrow_buffer::ScalarBuffer::from(vec![0, 2]),
        arrow_buffer::ScalarBuffer::from(vec![2, 2]),
        Arc::new(st2.clone()),
        None,
    );
    let fixed = arrow_array::FixedSizeListArray::new(Arc::clone(&item), 2, Arc::new(st2), None);
    let s2 = schema(vec![
        f("v", view.data_type().clone()),
        f("fx", fixed.data_type().clone()),
    ]);
    let batch =
        RecordBatch::try_new(Arc::clone(&s2), vec![Arc::new(view), Arc::new(fixed)]).unwrap();
    let paths = [
        ColumnPath::nested(["v", "y"]),
        ColumnPath::nested(["fx", "x"]),
    ];
    let out = project_batch(&batch, &paths).unwrap();
    assert_eq!(out.schema(), project_schema(&s2, &paths).unwrap());
}

fn inner2() -> Fields {
    Fields::from(vec![f("x", DataType::Int32), f("y", DataType::Int32)])
}

// ---- statistics ------------------------------------------------------------

fn stats(rows: u64, nulls: u64, min: ScalarValue, max: ScalarValue) -> FileStatistics {
    FileStatistics {
        rows: Precision::Exact(rows),
        bytes: Precision::Absent,
        columns: vec![ColumnStatistics {
            path: col("c"),
            null_count: Precision::Exact(nulls),
            min: Precision::Exact(min),
            max: Precision::Exact(max),
            nan_count: Precision::Absent,
        }],
    }
}

fn cmp(op: CmpOp, v: i32) -> PushdownExpr {
    PushdownExpr::cmp(col("c"), op, ScalarValue::Int32(v))
}

#[test]
fn float_bounds_without_a_nan_count_never_hide_nan_rows() {
    use TriState::{False as F, True as T, Unknown as U};
    // Parquet bounds leave NaN out: a unit {1.0, 2.0, NaN} has max 2.0, and
    // in Spark NaN > 5.0, NaN >= 5.0 and NaN <> 5.0 hold.
    let f = |op, v: f64| PushdownExpr::cmp(col("c"), op, ScalarValue::Float64(v));
    let mut s = stats(3, 0, ScalarValue::Float64(1.0), ScalarValue::Float64(2.0));
    for (op, v) in [(CmpOp::Gt, 5.0), (CmpOp::GtEq, 5.0)] {
        assert_eq!(prune(&f(op, v), &s), U, "{op:?} {v} may match NaN");
    }
    // {7.0, NaN}: bounds 7.0..7.0, yet NaN <> 7.0.
    let sevens = stats(2, 0, ScalarValue::Float64(7.0), ScalarValue::Float64(7.0));
    assert_eq!(prune(&f(CmpOp::NotEq, 7.0), &sevens), U);
    assert_eq!(prune(&f(CmpOp::NotEq, 9.0), &sevens), T);
    // Comparisons NaN fails still prune, but never claim every row.
    assert_eq!(prune(&f(CmpOp::Lt, 0.5), &s), F);
    assert_eq!(prune(&f(CmpOp::Eq, 7.0), &s), F);
    assert_eq!(prune(&f(CmpOp::Lt, 5.0), &s), U);
    assert_eq!(prune(&f(CmpOp::LtEq, 2.0), &s), U);
    // A unit known to hold no NaN prunes and concludes as usual.
    s.columns[0].nan_count = Precision::Exact(0);
    assert_eq!(prune(&f(CmpOp::Gt, 5.0), &s), F);
    assert_eq!(prune(&f(CmpOp::NotEq, 7.0), &s), T);
    let mut sevens = sevens;
    sevens.columns[0].nan_count = Precision::Exact(0);
    assert_eq!(prune(&f(CmpOp::NotEq, 7.0), &sevens), F);
    assert_eq!(prune(&f(CmpOp::Lt, 5.0), &s), T);
    s.columns[0].nan_count = Precision::Exact(1);
    assert_eq!(prune(&f(CmpOp::GtEq, 5.0), &s), U);
}

#[test]
fn pruning_answers_all_none_or_unknown() {
    use TriState::{False as F, True as T, Unknown as U};
    let s = stats(10, 0, ScalarValue::Int32(5), ScalarValue::Int32(9));
    let cases = [
        (CmpOp::Eq, 4, F),
        (CmpOp::Eq, 10, F),
        (CmpOp::Eq, 7, U),
        (CmpOp::NotEq, 4, T),
        (CmpOp::NotEq, 7, U),
        (CmpOp::Lt, 5, F),
        (CmpOp::Lt, 10, T),
        (CmpOp::Lt, 7, U),
        (CmpOp::LtEq, 4, F),
        (CmpOp::LtEq, 9, T),
        (CmpOp::Gt, 9, F),
        (CmpOp::Gt, 4, T),
        (CmpOp::GtEq, 10, F),
        (CmpOp::GtEq, 5, T),
    ];
    for (op, v, want) in cases {
        assert_eq!(prune(&cmp(op, v), &s), want, "{op:?} {v}");
    }
    let one = stats(3, 0, ScalarValue::Int32(5), ScalarValue::Int32(5));
    assert_eq!(prune(&cmp(CmpOp::Eq, 5), &one), T);
    assert_eq!(prune(&cmp(CmpOp::NotEq, 5), &one), F);
    // With nulls, "all rows" is never concluded.
    let nulls = stats(10, 2, ScalarValue::Int32(5), ScalarValue::Int32(9));
    assert_eq!(prune(&cmp(CmpOp::Lt, 10), &nulls), U);
    assert_eq!(prune(&PushdownExpr::IsNull(col("c")), &nulls), U);
    assert_eq!(prune(&PushdownExpr::IsNull(col("c")), &s), F);
    assert_eq!(prune(&PushdownExpr::IsNotNull(col("c")), &s), T);
    let all_null = stats(4, 4, ScalarValue::Int32(0), ScalarValue::Int32(0));
    assert_eq!(prune(&cmp(CmpOp::Eq, 0), &all_null), F);
    assert_eq!(prune(&PushdownExpr::IsNull(col("c")), &all_null), T);
    assert_eq!(prune(&PushdownExpr::IsNotNull(col("c")), &all_null), F);
    // IN, AND, OR, NOT, constants.
    let inn = |vals: &[i32]| PushdownExpr::In {
        column: col("c"),
        values: vals.iter().map(|v| ScalarValue::Int32(*v)).collect(),
        collation: Collation::Utf8Binary,
    };
    assert_eq!(prune(&inn(&[1, 2]), &s), F);
    assert_eq!(prune(&inn(&[1, 7]), &s), U);
    assert_eq!(prune(&inn(&[5]), &one), T);
    assert_eq!(
        prune(
            &PushdownExpr::And(vec![cmp(CmpOp::Gt, 4), cmp(CmpOp::Lt, 10)]),
            &s
        ),
        T
    );
    assert_eq!(
        prune(
            &PushdownExpr::And(vec![cmp(CmpOp::Gt, 4), cmp(CmpOp::Lt, 7)]),
            &s
        ),
        U
    );
    assert_eq!(
        prune(
            &PushdownExpr::Or(vec![cmp(CmpOp::Gt, 9), cmp(CmpOp::Lt, 5)]),
            &s
        ),
        F
    );
    assert_eq!(
        prune(
            &PushdownExpr::Or(vec![cmp(CmpOp::Gt, 9), cmp(CmpOp::Lt, 7)]),
            &s
        ),
        U
    );
    assert_eq!(
        prune(
            &PushdownExpr::Or(vec![cmp(CmpOp::Gt, 9), cmp(CmpOp::Lt, 10)]),
            &s
        ),
        T
    );
    assert_eq!(
        prune(&PushdownExpr::Not(Box::new(cmp(CmpOp::Lt, 5))), &s),
        T
    );
    assert_eq!(prune(&PushdownExpr::Constant(false), &s), F);
    // Unknown column, other collations, inexact bounds, unordered floats.
    assert_eq!(
        prune(
            &PushdownExpr::cmp(col("x"), CmpOp::Eq, ScalarValue::Int32(1)),
            &s
        ),
        U
    );
    assert_eq!(prune(&PushdownExpr::IsNull(col("x")), &s), U);
    let lcase = Collation::Other("UTF8_LCASE".into());
    assert_eq!(
        prune(
            &PushdownExpr::Cmp {
                column: col("c"),
                op: CmpOp::Eq,
                value: ScalarValue::Int32(1),
                collation: lcase.clone()
            },
            &s
        ),
        U
    );
    assert_eq!(
        prune(
            &PushdownExpr::In {
                column: col("c"),
                values: vec![],
                collation: lcase.clone()
            },
            &s
        ),
        U
    );
    let mut inexact = s.clone();
    inexact.columns[0].min = inexact.columns[0].min.clone().to_inexact();
    inexact.columns[0].max = inexact.columns[0].max.clone().to_inexact();
    assert_eq!(prune(&cmp(CmpOp::Eq, 100), &inexact), U);
    let fl = stats(
        5,
        0,
        ScalarValue::Float64(-0.0),
        ScalarValue::Float64(f64::NAN),
    );
    let fcmp = |op, v: f64| PushdownExpr::cmp(col("c"), op, ScalarValue::Float64(v));
    assert_eq!(prune(&fcmp(CmpOp::Lt, -1.0), &fl), U);
    assert_eq!(prune(&fcmp(CmpOp::Gt, 1e300), &fl), U);
    let fl2 = stats(5, 0, ScalarValue::Float64(1.0), ScalarValue::Float64(2.0));
    assert_eq!(prune(&fcmp(CmpOp::Eq, f64::NAN), &fl2), U);
    assert_eq!(prune(&fcmp(CmpOp::Eq, 0.0), &fl2), U);
    assert_eq!(prune(&fcmp(CmpOp::Eq, 3.0), &fl2), F);
    // Strings and prefixes.
    let st = stats(
        5,
        0,
        ScalarValue::Utf8("apple".into()),
        ScalarValue::Utf8("banana".into()),
    );
    let sw = |p: &str, c: Collation| PushdownExpr::StartsWith {
        column: col("c"),
        prefix: p.into(),
        collation: c,
    };
    assert_eq!(prune(&sw("c", Collation::Utf8Binary), &st), F);
    assert_eq!(prune(&sw("aa", Collation::Utf8Binary), &st), F);
    assert_eq!(prune(&sw("b", Collation::Utf8Binary), &st), U);
    assert_eq!(prune(&sw("b", lcase), &st), U);
    let same = stats(
        5,
        0,
        ScalarValue::Utf8("bx".into()),
        ScalarValue::Utf8("by".into()),
    );
    assert_eq!(prune(&sw("b", Collation::Utf8Binary), &same), T);
    // Statistics cannot rule a substring in or out.
    let has = PushdownExpr::Contains {
        column: col("c"),
        needle: "zz".into(),
        collation: Collation::Utf8Binary,
    };
    assert_eq!(prune(&has, &st), U);
    assert_eq!(prune(&PushdownExpr::Not(Box::new(has)).to_nnf(), &st), U);
    assert_eq!(prune(&sw("b", Collation::Utf8Binary), &all_null), F);
    assert_eq!(prune(&sw("b", Collation::Utf8Binary), &s), U);
    // Every comparable type.
    for (lo, hi, v) in [
        (
            ScalarValue::Boolean(false),
            ScalarValue::Boolean(false),
            ScalarValue::Boolean(true),
        ),
        (
            ScalarValue::Int8(1),
            ScalarValue::Int8(2),
            ScalarValue::Int8(3),
        ),
        (
            ScalarValue::Int16(1),
            ScalarValue::Int16(2),
            ScalarValue::Int16(3),
        ),
        (
            ScalarValue::Int64(1),
            ScalarValue::Int64(2),
            ScalarValue::Int64(3),
        ),
        (
            ScalarValue::Float32(1.0),
            ScalarValue::Float32(2.0),
            ScalarValue::Float32(3.0),
        ),
        (
            ScalarValue::Date32(1),
            ScalarValue::Date32(2),
            ScalarValue::Date32(3),
        ),
        (
            ScalarValue::TimestampMicros(1),
            ScalarValue::TimestampMicros(2),
            ScalarValue::TimestampMicros(3),
        ),
        (
            ScalarValue::Binary(vec![1]),
            ScalarValue::Binary(vec![2]),
            ScalarValue::Binary(vec![3]),
        ),
        (
            ScalarValue::Decimal128 {
                unscaled: 1,
                precision: 5,
                scale: 2,
            },
            ScalarValue::Decimal128 {
                unscaled: 2,
                precision: 5,
                scale: 2,
            },
            ScalarValue::Decimal128 {
                unscaled: 3,
                precision: 5,
                scale: 2,
            },
        ),
    ] {
        let s = stats(2, 0, lo, hi);
        assert_eq!(prune(&PushdownExpr::eq(col("c"), v), &s), F);
    }
    // Different decimal scales are not compared.
    let d = stats(
        2,
        0,
        ScalarValue::Decimal128 {
            unscaled: 1,
            precision: 5,
            scale: 2,
        },
        ScalarValue::Decimal128 {
            unscaled: 2,
            precision: 5,
            scale: 2,
        },
    );
    assert_eq!(
        prune(
            &PushdownExpr::eq(
                col("c"),
                ScalarValue::Decimal128 {
                    unscaled: 9,
                    precision: 5,
                    scale: 1
                }
            ),
            &d
        ),
        U
    );
    assert_eq!(Precision::<u8>::Absent.to_inexact(), Precision::Absent);
    assert_eq!(Precision::Inexact(3).value(), Some(&3));
    assert_ne!(ScalarValue::Float64(1.0), ScalarValue::Int32(1));
    assert_eq!(ScalarValue::Null, ScalarValue::Null);
    assert_eq!(
        ScalarValue::Float32(f32::NAN),
        ScalarValue::Float32(f32::NAN)
    );
}

// ---- options ---------------------------------------------------------------

static OPTS: OptionSchema = OptionSchema {
    format: "t",
    options: &[
        OptionSpec {
            name: "flag",
            kind: OptionKind::Bool,
            default: OptionDefault::Bool(false),
            conf_fallback: Some("spark.sql.t.flag"),
            metadata_key: false,
            doc: "",
        },
        OptionSpec {
            name: "level",
            kind: OptionKind::Int { min: 0, max: 9 },
            default: OptionDefault::Int(3),
            conf_fallback: Some("krishiv.io.maxRetries"),
            metadata_key: true,
            doc: "",
        },
        OptionSpec {
            name: "mode",
            kind: OptionKind::Enum(&["FAST", "SAFE"]),
            default: OptionDefault::Str("SAFE"),
            conf_fallback: None,
            metadata_key: false,
            doc: "",
        },
        OptionSpec {
            name: "comment",
            kind: OptionKind::String,
            default: OptionDefault::None,
            conf_fallback: None,
            metadata_key: false,
            doc: "",
        },
        OptionSpec {
            name: "key",
            kind: OptionKind::Secret,
            default: OptionDefault::None,
            conf_fallback: None,
            metadata_key: true,
            doc: "",
        },
    ],
};

#[derive(Debug)]
struct Vault;

impl SecretResolver for Vault {
    fn resolve<'a>(&'a self, scope: &'a str, key: &'a str) -> BoxFuture<'a, Result<SecretString>> {
        Box::pin(async move { Ok(SecretString::new(format!("{scope}/{key}"))) })
    }
}

fn parse(pairs: &[(&str, OptionValue)], c: &Config) -> Result<(FormatOptions, Vec<OptionWarning>)> {
    let user: BTreeMap<String, OptionValue> = pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    OPTS.parse(&user, c)
}

#[test]
fn options_are_typed_once() {
    let plain = |k: &str, v: &str| (k.to_owned(), OptionValue::from_user(k, v));
    let (c, _) =
        Config::from_pairs([("spark.sql.t.flag", "true"), ("krishiv.io.maxRetries", "7")]).unwrap();
    let (o, w) = parse(&[], &c).unwrap();
    assert!(w.is_empty());
    assert_eq!(
        (o.bool("FLAG"), o.int("level"), o.str("mode")),
        (Some(true), Some(7), Some("SAFE"))
    );
    let pairs: Vec<(String, OptionValue)> = vec![
        plain("Mode", "fast"),
        plain("comment", " hi "),
        plain("levelx", "1"),
    ];
    let user: BTreeMap<String, OptionValue> = pairs.into_iter().collect();
    let (o, w) = OPTS.parse(&user, &Config::new()).unwrap();
    assert_eq!(
        (o.str("mode"), o.str("comment")),
        (Some("FAST"), Some(" hi "))
    );
    assert_eq!(
        w,
        vec![OptionWarning::Unknown {
            key: "levelx".into(),
            suggestion: Some("level")
        }]
    );
    assert_eq!(
        w[0].to_string(),
        "option levelx is not known and is ignored; did you mean level?"
    );
    assert_eq!(
        OptionWarning::Unknown {
            key: "zzzzzz".into(),
            suggestion: None
        }
        .to_string(),
        "option zzzzzz is not known and is ignored"
    );
    assert!(o.int("comment").is_none() && o.bool("mode").is_none() && o.str("nope").is_none());
    assert_eq!(o.schema().format, "t");
    for (k, v) in [
        ("flag", "maybe"),
        ("level", "12"),
        ("level", "x"),
        ("mode", "slow"),
    ] {
        let e = parse(&[(k, OptionValue::from_user(k, v))], &Config::new()).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivInvalidFormatOptionValue,
            "{k}={v}"
        );
    }
    let e = parse(
        &[("flag", OptionValue::reference("s", "k"))],
        &Config::new(),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivInvalidFormatOptionValue);
}

#[test]
fn strict_mode_rejects_unknown_options() {
    let (strict, _) = Config::from_pairs([("krishiv.io.strictOptions", "true")]).unwrap();
    let e = parse(&[("modde", OptionValue::from_user("modde", "x"))], &strict).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnknownFormatOption);
    assert!(e.to_string().contains("Did you mean \"mode\"?"));
    let e = parse(&[("zzzzzzzz", OptionValue::from_user("z", "x"))], &strict).unwrap_err();
    assert!(!e.to_string().contains("Did you mean"));
    // Framework and object-store options are never unknown.
    assert!(parse(&[("PATH", OptionValue::from_user("path", "/x"))], &strict).is_ok());
    let (o, w) = parse(
        &[
            (
                "fs.s3a.endpoint",
                OptionValue::from_user("fs.s3a.endpoint", "http://127.0.0.1:9000"),
            ),
            (
                "FS.S3A.SECRET.KEY",
                OptionValue::from_user("FS.S3A.SECRET.KEY", "hunter2"),
            ),
        ],
        &strict,
    )
    .unwrap();
    assert!(w.is_empty());
    assert!(o.ignored().is_empty());
}

#[test]
fn scan_options_read_the_session() {
    let (c, _) = Config::from_pairs([
        ("krishiv.io.prefetchBytes", "2m"),
        ("spark.sql.files.ignoreCorruptFiles", "true"),
        ("spark.sql.files.ignoreMissingFiles", "TRUE"),
    ])
    .unwrap();
    let o = ScanOptions::from_config(&c);
    assert_eq!(o.prefetch_bytes, 2 << 20);
    assert!(o.ignore_corrupt_files && o.ignore_missing_files);
    assert_eq!(ScanOptions::default().prefetch_bytes, 64 << 20);
}

#[test]
fn secrets_stay_secret_and_key_the_cache() {
    let rt = || OptionValue::from_user("key", "hunter2");
    let (o, _) = parse(&[("key", rt())], &Config::new()).unwrap();
    assert!(!format!("{o:?}").contains("hunter2"));
    assert!(o.has_secret("key") && !o.has_secret("flag"));
    let got = o
        .resolve_secret("key", &Vault)
        .now_or_never()
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(got.expose_secret(), "hunter2");
    assert!(
        o.resolve_secret("flag", &Vault)
            .now_or_never()
            .unwrap()
            .unwrap()
            .is_none()
    );
    assert!(
        o.resolve_secret("nope", &Vault)
            .now_or_never()
            .unwrap()
            .unwrap()
            .is_none()
    );
    let fp1 = o
        .metadata_fingerprint(&Vault)
        .now_or_never()
        .unwrap()
        .unwrap();
    let (o2, _) = parse(
        &[("key", OptionValue::Plain("other".into()))],
        &Config::new(),
    )
    .unwrap();
    let fp2 = o2
        .metadata_fingerprint(&Vault)
        .now_or_never()
        .unwrap()
        .unwrap();
    assert!(fp1.is_some() && fp1 != fp2);
    // A reference resolves through the provider; without one it fails.
    let (o3, _) = parse(
        &[("key", OptionValue::reference("scope", "name"))],
        &Config::new(),
    )
    .unwrap();
    let got = o3
        .resolve_secret("key", &Vault)
        .now_or_never()
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(got.expose_secret(), "scope/name");
    let e = o3
        .resolve_secret("key", &options::NoSecrets)
        .now_or_never()
        .unwrap()
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // A URL-shaped secret is still a secret.
    let (o4, _) = parse(
        &[("key", OptionValue::from_user("url", "s3://u:pw@b/k"))],
        &Config::new(),
    )
    .unwrap();
    assert!(!format!("{o4:?}").contains("pw@"));
    // Defaults only: no fingerprintable key unless one is set (level is a
    // metadata key with a default).
    let d = FormatOptions::defaults(&OPTS);
    assert!(
        d.metadata_fingerprint(&Vault)
            .now_or_never()
            .unwrap()
            .unwrap()
            .is_some()
    );
    assert!(d.ignored().is_empty());
}

// ---- schema mapping --------------------------------------------------------

fn mapper(
    table: &SchemaRef,
    file: &SchemaRef,
    mapping: FieldMapping,
) -> std::result::Result<SchemaMapper, KrishivError> {
    let mut req = ScanRequest::all(Arc::clone(table));
    req.mapping = mapping;
    SchemaMapper::new(&req, Arc::clone(file), "memory://x/f")
}

fn by_name() -> FieldMapping {
    FieldMapping::ByName {
        case_sensitive: false,
    }
}

#[test]
fn leaf_types_widen_narrow_and_change_representation() {
    let file = schema(vec![
        f("i8", DataType::Int8),
        f("i32", DataType::Int32),
        f("f32", DataType::Float32),
        f("d", DataType::Decimal128(5, 2)),
        f("date", DataType::Date32),
        f("s", DataType::Utf8),
        f("ls", DataType::LargeUtf8),
        f("b", DataType::Binary),
        f(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("+01:00".into())),
        ),
        f("n", DataType::Int32),
    ]);
    let table = schema(vec![
        f("i8", DataType::Decimal128(10, 2)),
        f("i32", DataType::Float64),
        f("f32", DataType::Float64),
        f("d", DataType::Decimal128(10, 4)),
        f("date", DataType::Timestamp(TimeUnit::Microsecond, None)),
        f("s", DataType::Utf8View),
        f("ls", DataType::Utf8),
        f("b", DataType::BinaryView),
        f(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        ),
        f("n", DataType::Int16),
    ]);
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int8Array::from(vec![Some(-3), None])),
        Arc::new(Int32Array::from(vec![Some(7), None])),
        Arc::new(Float32Array::from(vec![Some(1.5), None])),
        Arc::new(
            Decimal128Array::from(vec![Some(123), None])
                .with_precision_and_scale(5, 2)
                .unwrap(),
        ),
        Arc::new(Date32Array::from(vec![Some(1), None])),
        Arc::new(StringArray::from(vec![Some("a"), None])),
        Arc::new(LargeStringArray::from(vec![Some("b"), None])),
        Arc::new(BinaryArray::from(vec![Some(&b"c"[..]), None])),
        Arc::new(
            arrow_array::TimestampMicrosecondArray::from(vec![Some(5), None])
                .with_timezone("+01:00"),
        ),
        Arc::new(Int32Array::from(vec![Some(300), None])),
    ];
    let m = mapper(&table, &file, by_name()).unwrap();
    let out = m
        .map_batch(&RecordBatch::try_new(Arc::clone(&file), cols).unwrap())
        .unwrap();
    assert_eq!(
        out.column(0).as_primitive::<Decimal128Type>().value(0),
        -300
    );
    assert_eq!(out.column(1).as_primitive::<Float64Type>().value(0), 7.0);
    assert_eq!(out.column(2).as_primitive::<Float64Type>().value(0), 1.5);
    assert_eq!(
        out.column(3).as_primitive::<Decimal128Type>().value(0),
        12300
    );
    assert_eq!(
        out.column(4)
            .as_primitive::<TimestampMicrosecondType>()
            .value(0),
        86_400_000_000
    );
    assert_eq!(out.column(5).as_string_view().value(0), "a");
    assert_eq!(out.column(6).as_string::<i32>().value(0), "b");
    assert_eq!(out.column(7).as_binary_view().value(0), b"c");
    assert_eq!(out.column(9).as_primitive::<Int16Type>().value(0), 300);
    assert!((0..10).all(|c| out.column(c).is_null(1)));
    // Narrowing overflow is an error, never a wrapped value.
    let big = RecordBatch::try_new(
        schema(vec![f("n", DataType::Int32)]),
        vec![Arc::new(Int32Array::from(vec![70_000]))],
    )
    .unwrap();
    let m = mapper(
        &schema(vec![f("n", DataType::Int16)]),
        &big.schema(),
        by_name(),
    )
    .unwrap();
    assert_eq!(
        m.map_batch(&big).unwrap_err().error_class(),
        ErrorClass::CastOverflow
    );
    // A corrupt decimal beyond its declared precision fails, never overflows.
    let dec = |p, s| DataType::Decimal128(p, s);
    let corrupt = RecordBatch::try_new(
        schema(vec![f("d", dec(5, 0))]),
        vec![Arc::new(
            Decimal128Array::from(vec![i128::MAX / 10])
                .with_precision_and_scale(5, 0)
                .unwrap(),
        )],
    )
    .unwrap();
    let m = mapper(
        &schema(vec![f("d", dec(38, 33))]),
        &corrupt.schema(),
        by_name(),
    )
    .unwrap();
    assert_eq!(
        m.map_batch(&corrupt).unwrap_err().error_class(),
        ErrorClass::CastOverflow
    );
    // Illegal changes.
    for (fdt, tdt) in [
        (DataType::Utf8, DataType::Int64),
        (DataType::Int64, DataType::Int32),
        (DataType::Float64, DataType::Float32),
        (DataType::Decimal128(10, 2), DataType::Decimal128(10, 3)),
        (DataType::Int64, DataType::Decimal128(10, 0)),
        (
            DataType::Int32,
            DataType::Struct(Fields::from(vec![f("x", DataType::Int32)])),
        ),
    ] {
        let e = mapper(
            &schema(vec![f("c", tdt.clone())]),
            &schema(vec![f("c", fdt.clone())]),
            by_name(),
        )
        .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::FailedReadFileParquetColumnDataTypeMismatch,
            "{fdt} -> {tdt}"
        );
        assert!(e.to_string().contains("[c]"));
    }
    // A wrong decoder output is an internal error.
    let m = mapper(
        &schema(vec![f("a", DataType::Int32)]),
        &schema(vec![f("a", DataType::Int32)]),
        by_name(),
    )
    .unwrap();
    let bad = RecordBatch::new_empty(schema(vec![]));
    assert_eq!(
        m.map_batch(&bad).unwrap_err().error_class(),
        ErrorClass::InternalError
    );
}

#[test]
fn nested_types_map_recursively() {
    let fst = Fields::from(vec![f("x", DataType::Int32), f("w", DataType::Utf8)]);
    let tst = Fields::from(vec![f("X", DataType::Int64), f("y", DataType::Utf8View)]);
    let fentries = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            f("value", DataType::Int32),
        ])),
        false,
    );
    let tentries = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8View, false),
            f("value", DataType::Int64),
        ])),
        false,
    );
    let file = schema(vec![
        f("st", DataType::Struct(fst.clone())),
        f(
            "l",
            DataType::List(Arc::new(f("item", DataType::Struct(fst.clone())))),
        ),
        f("m", DataType::Map(Arc::new(fentries.clone()), false)),
    ]);
    let table = schema(vec![
        f("st", DataType::Struct(tst.clone())),
        f(
            "l",
            DataType::List(Arc::new(f("item", DataType::Struct(tst.clone())))),
        ),
        f("m", DataType::Map(Arc::new(tentries), false)),
        f("gone", DataType::Struct(tst)),
    ]);
    let m = mapper(&table, &file, by_name()).unwrap();
    // Only `st.x` of the struct is read; lists and maps whole.
    let fr = m.file_request(&ScanRequest::all(Arc::clone(&table)));
    assert_eq!(
        fr.projection,
        vec![ColumnPath::nested(["st", "x"]), col("l"), col("m")]
    );
    let st_arr = |n: usize| -> StructArray {
        StructArray::try_new(
            Fields::from(vec![f("x", DataType::Int32)]),
            vec![Arc::new(Int32Array::from(
                (0..n)
                    .map(|i| i32::try_from(i).unwrap())
                    .collect::<Vec<_>>(),
            ))],
            None,
        )
        .unwrap()
    };
    let full_st = |n: usize| -> StructArray {
        StructArray::try_new(
            fst.clone(),
            vec![
                Arc::new(Int32Array::from(
                    (0..n)
                        .map(|i| i32::try_from(i).unwrap())
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(vec!["w"; n])),
            ],
            None,
        )
        .unwrap()
    };
    let list = ListArray::try_new(
        Arc::new(f("item", DataType::Struct(fst.clone()))),
        OffsetBuffer::from_lengths([1, 2]),
        Arc::new(full_st(3)),
        None,
    )
    .unwrap();
    let DataType::Struct(kv) = fentries.data_type() else {
        panic!()
    };
    let entries = StructArray::try_new(
        kv.clone(),
        vec![
            Arc::new(StringArray::from(vec!["k1", "k2"])),
            Arc::new(Int32Array::from(vec![1, 2])),
        ],
        None,
    )
    .unwrap();
    let map = MapArray::try_new(
        Arc::new(fentries),
        OffsetBuffer::from_lengths([2, 0]),
        entries,
        None,
        false,
    )
    .unwrap();
    let batch = RecordBatch::try_new(
        Arc::clone(m.file_output_schema()),
        vec![Arc::new(st_arr(2)), Arc::new(list), Arc::new(map)],
    )
    .unwrap();
    let out = m.map_batch(&batch).unwrap();
    let st = out.column(0).as_struct();
    assert_eq!(st.column(0).as_primitive::<Int64Type>().value(1), 1);
    assert_eq!(st.column(1).null_count(), 2);
    let l = out.column(1).as_list::<i32>();
    assert_eq!(
        l.value(1)
            .as_struct()
            .column(0)
            .as_primitive::<Int64Type>()
            .value(1),
        2
    );
    let mp = out.column(2).as_map();
    assert_eq!(mp.keys().as_string_view().value(1), "k2");
    assert_eq!(out.column(3).null_count(), 2);
    assert!(out.schema().fields().iter().all(|f| f.is_nullable()));
    // Mismatched nested types.
    let bad = schema(vec![f(
        "l",
        DataType::List(Arc::new(f("item", DataType::Utf8))),
    )]);
    let flist = schema(vec![f(
        "l",
        DataType::List(Arc::new(f("item", DataType::Int32))),
    )]);
    assert!(mapper(&bad, &flist, by_name()).is_err());
}

#[test]
fn a_struct_without_the_requested_fields_keeps_its_nulls() {
    let fst = Fields::from(vec![f("x", DataType::Int32), f("w", DataType::Utf8)]);
    let file = schema(vec![f("st", DataType::Struct(fst))]);
    let table = schema(vec![f(
        "st",
        DataType::Struct(Fields::from(vec![f("y", DataType::Int64)])),
    )]);
    let m = mapper(&table, &file, by_name()).unwrap();
    let fr = m.file_request(&ScanRequest::all(Arc::clone(&table)));
    assert_eq!(fr.projection, vec![ColumnPath::nested(["st", "x"])]);
    let stored = StructArray::try_new(
        Fields::from(vec![f("x", DataType::Int32)]),
        vec![Arc::new(Int32Array::from(vec![Some(1), None]))],
        Some(arrow_buffer::NullBuffer::from(vec![true, false])),
    )
    .unwrap();
    let out = m
        .map_batch(
            &RecordBatch::try_new(Arc::clone(m.file_output_schema()), vec![Arc::new(stored)])
                .unwrap(),
        )
        .unwrap();
    let st = out.column(0).as_struct();
    assert!(
        st.is_valid(0) && st.is_null(1),
        "the stored struct's validity is kept"
    );
    assert_eq!(st.column(0).null_count(), 2);
}

#[test]
fn ntz_and_ltz_read_as_each_other() {
    let ntz = DataType::Timestamp(TimeUnit::Microsecond, None);
    let ltz = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    for (file_t, table_t) in [(ntz.clone(), ltz.clone()), (ltz, ntz)] {
        let file = schema(vec![f("t", file_t.clone())]);
        let a: ArrayRef = match &file_t {
            DataType::Timestamp(_, tz) => Arc::new(
                arrow_array::TimestampMicrosecondArray::from(vec![7]).with_timezone_opt(tz.clone()),
            ),
            _ => unreachable!(),
        };
        let m = mapper(&schema(vec![f("t", table_t.clone())]), &file, by_name()).unwrap();
        let out = m
            .map_batch(&RecordBatch::try_new(Arc::clone(&file), vec![a]).unwrap())
            .unwrap();
        assert_eq!(out.column(0).data_type(), &table_t);
        assert_eq!(
            out.column(0)
                .as_primitive::<TimestampMicrosecondType>()
                .value(0),
            7
        );
        let lit = PushdownExpr::cmp(col("t"), CmpOp::Eq, ScalarValue::TimestampMicros(7));
        assert_eq!(m.map_predicate(&lit), Some(lit.clone()));
    }
}

#[test]
fn case_rules_field_ids_and_column_mapping() {
    let id = |n: &str, i: &str| {
        f(n, DataType::Int32)
            .with_metadata(HashMap::from([(FIELD_ID_KEY.to_owned(), i.to_owned())]))
    };
    let file = schema(vec![
        id("Alpha", "1"),
        id("beta", "2"),
        f("gamma", DataType::Int32),
    ]);
    let batch = RecordBatch::try_new(
        Arc::clone(&file),
        (1..=3)
            .map(|v| Arc::new(Int32Array::from(vec![v])) as ArrayRef)
            .collect(),
    )
    .unwrap();
    let read = |table: SchemaRef, mapping: FieldMapping| -> RecordBatch {
        let m = mapper(&table, &file, mapping).unwrap();
        let fr = m.file_request(&ScanRequest::all(table));
        m.map_batch(&project(&batch, &fr)).unwrap()
    };
    let one = |b: &RecordBatch, c: usize| -> Option<i32> {
        let a = b.column(c).as_primitive::<arrow_array::types::Int32Type>();
        a.is_valid(0).then(|| a.value(0))
    };
    let t = schema(vec![f("ALPHA", DataType::Int32)]);
    assert_eq!(one(&read(Arc::clone(&t), by_name()), 0), Some(1));
    assert_eq!(
        one(
            &read(
                t,
                FieldMapping::ByName {
                    case_sensitive: true
                }
            ),
            0
        ),
        None
    );
    let t = schema(vec![
        id("renamed", "2"),
        f("no_id", DataType::Int32),
        f("gamma", DataType::Int32),
    ]);
    let b = read(t, FieldMapping::FieldId);
    // Without an id: by name (Spark 4.2's fallback).
    assert_eq!(
        (one(&b, 0), one(&b, 1), one(&b, 2)),
        (Some(2), None, Some(3))
    );
    let phys = |n: &str, p: &str| {
        f(n, DataType::Int32).with_metadata(HashMap::from([(
            "delta.columnMapping.physicalName".to_owned(),
            p.to_owned(),
        )]))
    };
    let b = read(
        schema(vec![phys("logical", "gamma")]),
        FieldMapping::DeltaColumnMapping { by_id: false },
    );
    assert_eq!(one(&b, 0), Some(3));
    let did = f("x", DataType::Int32).with_metadata(HashMap::from([(
        "delta.columnMapping.id".to_owned(),
        "1".to_owned(),
    )]));
    assert_eq!(
        one(
            &read(
                schema(vec![did]),
                FieldMapping::DeltaColumnMapping { by_id: true }
            ),
            0
        ),
        Some(1)
    );
    let dup = schema(vec![f("a", DataType::Int32), f("A", DataType::Int32)]);
    let e = mapper(&schema(vec![f("a", DataType::Int32)]), &dup, by_name()).unwrap_err();
    // Spark 4.2: FAILED_READ_FILE.NO_HINT caused by _LEGACY_ERROR_TEMP_2093.
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    let cause = e.cause().unwrap();
    assert_eq!(cause.error_class(), ErrorClass::LegacyErrorTemp2093);
    assert_eq!(cause.message_parameters()[1].1, "[a, A]");
    assert_eq!(FieldMapping::default(), by_name());
}

fn project(batch: &RecordBatch, req: &ScanRequest) -> RecordBatch {
    let out = req.output_schema().unwrap();
    let cols = out
        .fields()
        .iter()
        .map(|f| Arc::clone(batch.column_by_name(f.name()).unwrap()))
        .collect();
    let opts = arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
    RecordBatch::try_new_with_options(out, cols, &opts).unwrap()
}

#[test]
fn predicates_are_retyped_to_the_file() {
    let file = schema(vec![
        f("i", DataType::Int32),
        f("f", DataType::Float32),
        f("d", DataType::Decimal128(5, 2)),
        f("dt", DataType::Date32),
        f("s", DataType::Utf8),
        f(
            "st",
            DataType::Struct(Fields::from(vec![f("x", DataType::Int32)])),
        ),
    ]);
    let table = schema(vec![
        f("i", DataType::Int64),
        f("f", DataType::Float64),
        f("d", DataType::Decimal128(10, 4)),
        f("dt", DataType::Timestamp(TimeUnit::Microsecond, None)),
        f("s", DataType::Utf8View),
        f(
            "st",
            DataType::Struct(Fields::from(vec![
                f("x", DataType::Int64),
                f("y", DataType::Int64),
            ])),
        ),
        f("z", DataType::Int64),
    ]);
    let m = mapper(&table, &file, by_name()).unwrap();
    let c = |n: &str, op, v| PushdownExpr::cmp(col(n), op, v);
    let map = |e: PushdownExpr| m.map_predicate(&e);
    let big = ScalarValue::Int64(3_000_000_000);
    assert_eq!(
        map(c("i", CmpOp::Eq, ScalarValue::Int64(5))),
        Some(c("i", CmpOp::Eq, ScalarValue::Int32(5)))
    );
    assert_eq!(
        map(c("i", CmpOp::Eq, big.clone())),
        Some(PushdownExpr::Constant(false))
    );
    assert_eq!(
        map(c("i", CmpOp::Lt, big.clone())),
        Some(PushdownExpr::IsNotNull(col("i")))
    );
    assert_eq!(
        map(c("i", CmpOp::Gt, ScalarValue::Int64(-3_000_000_000))),
        Some(PushdownExpr::IsNotNull(col("i")))
    );
    assert_eq!(
        map(c("i", CmpOp::LtEq, ScalarValue::Int64(-3_000_000_000))),
        Some(PushdownExpr::Constant(false))
    );
    assert_eq!(
        map(c("i", CmpOp::NotEq, big.clone())),
        Some(PushdownExpr::IsNotNull(col("i")))
    );
    assert_eq!(
        map(c("i", CmpOp::GtEq, ScalarValue::Float64(2.0))),
        Some(c("i", CmpOp::GtEq, ScalarValue::Int32(2)))
    );
    assert_eq!(
        map(c("i", CmpOp::Eq, ScalarValue::Float64(2.5))),
        Some(PushdownExpr::Constant(false))
    );
    assert_eq!(map(c("i", CmpOp::Lt, ScalarValue::Float64(2.5))), None);
    assert_eq!(map(c("i", CmpOp::Lt, ScalarValue::Float64(f64::NAN))), None);
    assert_eq!(
        map(c("f", CmpOp::Eq, ScalarValue::Float64(1.5))),
        Some(c("f", CmpOp::Eq, ScalarValue::Float32(1.5)))
    );
    assert_eq!(
        map(c("f", CmpOp::NotEq, ScalarValue::Float64(0.1))),
        Some(PushdownExpr::IsNotNull(col("f")))
    );
    assert_eq!(map(c("f", CmpOp::Lt, ScalarValue::Float64(f64::NAN))), None);
    let dec = |u: i128| ScalarValue::Decimal128 {
        unscaled: u,
        precision: 10,
        scale: 4,
    };
    assert_eq!(
        map(c("d", CmpOp::Eq, dec(12300))),
        Some(c(
            "d",
            CmpOp::Eq,
            ScalarValue::Decimal128 {
                unscaled: 123,
                precision: 5,
                scale: 2
            }
        ))
    );
    assert_eq!(
        map(c("d", CmpOp::Eq, dec(12345))),
        Some(PushdownExpr::Constant(false))
    );
    assert_eq!(map(c("d", CmpOp::Gt, dec(12345))), None);
    assert_eq!(
        map(c("d", CmpOp::Lt, dec(100_000_000))),
        Some(PushdownExpr::IsNotNull(col("d")))
    );
    assert_eq!(
        map(c("d", CmpOp::Lt, dec(-100_000_000))),
        Some(PushdownExpr::Constant(false))
    );
    const DAY: i64 = 86_400_000_000;
    assert_eq!(
        map(c("dt", CmpOp::Eq, ScalarValue::TimestampMicros(2 * DAY))),
        Some(c("dt", CmpOp::Eq, ScalarValue::Date32(2)))
    );
    assert_eq!(
        map(c("dt", CmpOp::Eq, ScalarValue::TimestampMicros(DAY + 1))),
        Some(PushdownExpr::Constant(false))
    );
    assert_eq!(
        map(c("s", CmpOp::Eq, ScalarValue::Utf8("a".into()))),
        Some(c("s", CmpOp::Eq, ScalarValue::Utf8("a".into())))
    );
    let sw = |n: &str| PushdownExpr::StartsWith {
        column: col(n),
        prefix: "p".into(),
        collation: Collation::Utf8Binary,
    };
    assert_eq!(map(sw("s")), Some(sw("s")));
    assert_eq!(map(sw("i")), None);
    assert_eq!(map(sw("z")), Some(PushdownExpr::Constant(false)));
    let has = |n: &str| PushdownExpr::Contains {
        column: col(n),
        needle: "p".into(),
        collation: Collation::Utf8Binary,
    };
    assert_eq!(map(has("s")), Some(has("s")));
    assert_eq!(has("s").columns(), [&col("s")]);
    assert_eq!(map(has("i")), None);
    assert_eq!(map(has("z")), Some(PushdownExpr::Constant(false)));
    // A negated substring stays a `Not` leaf.
    assert_eq!(
        map(PushdownExpr::Not(Box::new(has("s")))),
        Some(PushdownExpr::Not(Box::new(has("s"))))
    );
    // Zoned timestamps of another unit are not re-typed; of the same unit they are.
    let ts = |u, tz: &str| f("t", DataType::Timestamp(u, Some(tz.into())));
    let lit = c("t", CmpOp::Eq, ScalarValue::TimestampMicros(5));
    let other = mapper(
        &schema(vec![ts(TimeUnit::Microsecond, "UTC")]),
        &schema(vec![ts(TimeUnit::Millisecond, "UTC")]),
        by_name(),
    );
    assert!(other.is_err() || other.unwrap().map_predicate(&lit).is_none());
    let same = mapper(
        &schema(vec![ts(TimeUnit::Microsecond, "UTC")]),
        &schema(vec![ts(TimeUnit::Microsecond, "+01:00")]),
        by_name(),
    )
    .unwrap();
    assert_eq!(same.map_predicate(&lit), Some(lit.clone()));
    // Missing columns are all null.
    assert_eq!(
        map(c("z", CmpOp::Eq, ScalarValue::Int64(1))),
        Some(PushdownExpr::Constant(false))
    );
    assert_eq!(
        map(PushdownExpr::IsNull(col("z"))),
        Some(PushdownExpr::Constant(true))
    );
    assert_eq!(
        map(PushdownExpr::IsNotNull(col("z"))),
        Some(PushdownExpr::Constant(false))
    );
    assert_eq!(
        map(PushdownExpr::IsNull(col("i"))),
        Some(PushdownExpr::IsNull(col("i")))
    );
    // Nested paths.
    assert_eq!(
        map(c("st.x", CmpOp::Eq, ScalarValue::Int64(1))),
        None,
        "dotted names are not paths"
    );
    let nested = PushdownExpr::cmp(
        ColumnPath::nested(["st", "x"]),
        CmpOp::Eq,
        ScalarValue::Int64(1),
    );
    assert_eq!(
        map(nested),
        Some(PushdownExpr::cmp(
            ColumnPath::nested(["st", "x"]),
            CmpOp::Eq,
            ScalarValue::Int32(1)
        ))
    );
    assert_eq!(
        map(PushdownExpr::IsNull(ColumnPath::nested(["st", "y"]))),
        Some(PushdownExpr::Constant(true))
    );
    assert_eq!(
        map(PushdownExpr::IsNull(ColumnPath::nested(["i", "x"]))),
        None
    );
    // IN: out-of-range values dropped; all out of range: false.
    let inn = |vals: Vec<ScalarValue>| PushdownExpr::In {
        column: col("i"),
        values: vals,
        collation: Collation::Utf8Binary,
    };
    assert_eq!(
        map(inn(vec![ScalarValue::Int64(1), big.clone()])),
        Some(PushdownExpr::In {
            column: col("i"),
            values: vec![ScalarValue::Int32(1)],
            collation: Collation::Utf8Binary
        })
    );
    assert_eq!(
        map(inn(vec![big.clone()])),
        Some(PushdownExpr::Constant(false))
    );
    assert_eq!(
        map(PushdownExpr::In {
            column: col("z"),
            values: vec![],
            collation: Collation::Utf8Binary
        }),
        Some(PushdownExpr::Constant(false))
    );
    assert_eq!(
        map(PushdownExpr::In {
            column: col("i"),
            values: vec![ScalarValue::Utf8("x".into())],
            collation: Collation::Utf8Binary
        }),
        None
    );
    // AND keeps what it can; OR needs every side.
    let unknown = c("i", CmpOp::Lt, ScalarValue::Float64(2.5));
    assert_eq!(
        map(PushdownExpr::And(vec![
            unknown.clone(),
            c("i", CmpOp::Eq, ScalarValue::Int64(1))
        ])),
        Some(PushdownExpr::And(vec![c(
            "i",
            CmpOp::Eq,
            ScalarValue::Int32(1)
        )]))
    );
    assert_eq!(map(PushdownExpr::And(vec![unknown.clone()])), None);
    assert_eq!(
        map(PushdownExpr::Or(vec![
            unknown,
            c("i", CmpOp::Eq, ScalarValue::Int64(1))
        ])),
        None
    );
    assert_eq!(
        map(PushdownExpr::Not(Box::new(sw("s")))),
        Some(PushdownExpr::Not(Box::new(sw("s"))))
    );
    // Runtime filters lower to the file's column and type when read.
    let filter = Arc::new(crate::RuntimeFilter::new(
        crate::FilterId(1),
        col("i"),
        DataType::Int64,
    ));
    let mut b = crate::KeyFilterBuilder::new(DataType::Int64, crate::KeyFilterOptions::default())
        .unwrap_or_else(|| panic!("int64 keys"));
    b.add(&arrow_array::Int64Array::from(vec![9, 5]), None)
        .unwrap_or_else(|e| panic!("{e}"));
    let keys = Arc::new(b.finish());
    let mut req = ScanRequest::all(Arc::clone(&table));
    req.dynamic
        .push(crate::ScanFilter::new(Arc::clone(&filter), &table).unwrap_or_else(|| panic!("i")));
    req.predicate = Some(c("i", CmpOp::Eq, big));
    let fr = m.file_request(&req);
    assert_eq!(fr.dynamic[0].predicate(), None, "pending");
    filter.publish(crate::FilterSnapshot::Final(Arc::clone(&keys)));
    let membership = PushdownExpr::Membership {
        column: col("i"),
        filter: crate::MembershipFilter {
            id: crate::FilterId(1),
            generation: 1,
            keys,
        },
    };
    assert_eq!(
        fr.dynamic[0].predicate(),
        Some(PushdownExpr::And(vec![
            c("i", CmpOp::GtEq, ScalarValue::Int32(5)),
            c("i", CmpOp::LtEq, ScalarValue::Int32(9)),
            PushdownExpr::In {
                column: col("i"),
                values: vec![ScalarValue::Int32(5), ScalarValue::Int32(9)],
                collation: Collation::Utf8Binary,
            },
            membership.clone(),
        ]))
    );
    // A membership test maps like a comparison.
    assert_eq!(map(membership.clone()), Some(membership));
    assert_eq!(fr.predicate, Some(PushdownExpr::Constant(false)));
}

// ---- formats and errors ----------------------------------------------------

#[test]
fn capabilities_and_metadata_needs() {
    let mut caps = Capabilities {
        splittable: Splittable::ByCodec,
        self_describing: false,
        statistics: false,
        row_index: false,
        selection: false,
        page_pruning: false,
        page_checksums: false,
        write: false,
    };
    assert!(
        caps.is_splittable("a.csv")
            && caps.is_splittable("a.csv.bz2")
            && !caps.is_splittable("a.csv.gz")
    );
    caps.splittable = Splittable::No;
    assert!(!caps.is_splittable("a"));
    caps.splittable = Splittable::Yes;
    assert!(caps.is_splittable("a.gz"));
    let none = MetadataNeeds::default();
    let all = MetadataNeeds {
        page_index: true,
        bloom: true,
    };
    assert!(all.covers(&none) && !none.covers(&all));
    assert_eq!(none.union(all), all);
    assert_eq!(spark_file_name(3, "u", 12, ".csv"), "part-00003-u-c012.csv");
}

#[test]
fn format_errors_get_the_path() {
    let p = "s3://b/k";
    let e = FormatError::CorruptFooter("x".into());
    assert_eq!(e.to_string(), "cannot read footer: x");
    assert_eq!(
        e.into_error(p).error_class(),
        ErrorClass::FailedReadFileCannotReadFileFooter
    );
    let e = FormatError::Corrupt("y".into());
    assert_eq!(e.to_string(), "corrupt data: y");
    let e = e.into_error(p);
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    assert!(e.to_string().contains("s3://b/k"));
    let t = FormatError::TypeMismatch {
        column: "[a]".into(),
        expected: "bigint".into(),
        actual: "BINARY".into(),
    };
    assert!(t.to_string().contains("expected bigint"));
    assert_eq!(
        t.into_error(p).error_class(),
        ErrorClass::FailedReadFileParquetColumnDataTypeMismatch
    );
    let o = FormatError::from(crate::error::internal("z"));
    assert!(o.to_string().contains('z'));
    assert_eq!(o.into_error(p).error_class(), ErrorClass::InternalError);
    assert_eq!(
        crate::error::cancelled().error_class(),
        ErrorClass::OperationCanceled
    );
}

#[test]
fn the_async_bridge_turns_reads_into_need_data() {
    let reader = PushReadAt::new(10);
    let r = Arc::clone(&reader);
    let stream = futures::stream::once(async move {
        let a = r.read_at(0..4).await?;
        let b = r.read_at(2..3).await?;
        let schema = schema(vec![f("n", DataType::Int32)]);
        let v = i32::from(a[0]) + i32::from(b[0]);
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![v]))])
            .map_err(|e| FormatError::Corrupt(e.to_string()))
    });
    let mut d = AsyncDecoder::new(Box::pin(stream), Arc::clone(&reader));
    assert!(format!("{d:?}").contains("AsyncDecoder"));
    let Step::NeedData(r) = d.step().unwrap() else {
        panic!()
    };
    assert_eq!(r, vec![0..4]);
    // Stepping again without a push asks again.
    assert!(matches!(d.step().unwrap(), Step::NeedData(_)));
    d.push(vec![(0..4, bytes::Bytes::from_static(&[1, 2, 3, 4]))])
        .unwrap();
    // Served bytes are released; the next read is a new request.
    let Step::NeedData(r) = d.step().unwrap() else {
        panic!()
    };
    assert_eq!(r, vec![2..3]);
    d.push(vec![(0..4, bytes::Bytes::from_static(&[1, 2, 3, 4]))])
        .unwrap();
    assert_eq!(d.buffered_bytes(), 4);
    let Step::Batch(b) = d.step().unwrap() else {
        panic!()
    };
    assert_eq!(
        b.column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0),
        4
    );
    assert!(matches!(d.step().unwrap(), Step::Done));
    assert!(matches!(d.step().unwrap(), Step::Done));
    assert_eq!(d.metrics().bytes_pushed, 8);
    assert_eq!(reader.size(), 10);
    // A range read in parts is released once every byte was read.
    let pr = PushReadAt::new(10);
    let mut d2 = AsyncDecoder::new(Box::pin(futures::stream::pending()), Arc::clone(&pr));
    d2.push(vec![(0..6, bytes::Bytes::from_static(&[0, 1, 2, 3, 4, 5]))])
        .unwrap();
    assert_eq!(
        pr.read_at(0..2).now_or_never().unwrap().unwrap().as_ref(),
        &[0, 1]
    );
    assert_eq!(
        pr.read_at(1..4).now_or_never().unwrap().unwrap().as_ref(),
        &[1, 2, 3]
    );
    assert_eq!(d2.buffered_bytes(), 6);
    assert_eq!(
        pr.read_at(4..6).now_or_never().unwrap().unwrap().as_ref(),
        &[4, 5]
    );
    assert_eq!(d2.buffered_bytes(), 0);
    // Out-of-range reads fail; a stream waiting on something else is an error.
    assert!(reader.read_at(5..20).now_or_never().unwrap().is_err());
    let mut stuck = AsyncDecoder::new(Box::pin(futures::stream::pending()), PushReadAt::new(1));
    assert!(stuck.step().is_err());
    let mut failing = AsyncDecoder::new(
        Box::pin(futures::stream::once(async {
            Err(FormatError::Corrupt("c".into()))
        })),
        PushReadAt::new(1),
    );
    assert!(failing.step().is_err());
}

#[test]
fn the_async_bridge_rejects_pushes_of_the_wrong_length() {
    let reader = PushReadAt::new(10);
    let r = Arc::clone(&reader);
    let stream = futures::stream::once(async move {
        let a = r.read_at(0..4).await?;
        Err(FormatError::Corrupt(format!("read {} bytes", a.len())))
    });
    let mut d = AsyncDecoder::new(Box::pin(stream), reader);
    assert!(matches!(d.step().unwrap(), Step::NeedData(_)));
    // Too short: a later read would slice past the end. Never a panic.
    let short = d.push(vec![(0..4, bytes::Bytes::from_static(&[1, 2]))]);
    assert!(matches!(short, Err(FormatError::Other(_))));
    #[expect(
        clippy::reversed_empty_ranges,
        reason = "the malformed push under test"
    )]
    let reversed = d.push(vec![(4..0, bytes::Bytes::new())]);
    assert!(reversed.is_err());
    assert_eq!(d.buffered_bytes(), 0);
    assert!(matches!(d.step().unwrap(), Step::NeedData(_)));
}
