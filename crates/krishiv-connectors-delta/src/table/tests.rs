// SPDX-License-Identifier: Apache-2.0
//! The predicate translation and the partition-value parsing, on their own.

use delta_kernel::expressions::{Expression as Expr, Predicate as Pred};
use delta_kernel::schema::{StructField, StructType};
use krishiv_io::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue};

use super::predicate::to_kernel;
use super::*;

fn schema() -> StructType {
    StructType::try_new([
        StructField::nullable("id", DataType::LONG),
        StructField::nullable("p", DataType::STRING),
        StructField::nullable("f", DataType::DOUBLE),
        StructField::nullable("d", DataType::decimal(10, 2).unwrap()),
        StructField::nullable(
            "s",
            DataType::Struct(Box::new(
                StructType::try_new([StructField::nullable("x", DataType::INTEGER)]).unwrap(),
            )),
        ),
    ])
    .unwrap()
}

fn col(name: &str) -> ColumnPath {
    ColumnPath::new(name)
}

fn eq(c: &str, v: ScalarValue) -> PushdownExpr {
    PushdownExpr::eq(col(c), v)
}

#[test]
fn comparisons_null_tests_and_in_translate() {
    let s = schema();
    assert_eq!(
        to_kernel(&eq("id", ScalarValue::Int64(5)), &s),
        Some(Pred::eq(Expr::column(["id"]), Expr::literal(5_i64)))
    );
    assert_eq!(
        to_kernel(
            &PushdownExpr::cmp(col("p"), CmpOp::GtEq, ScalarValue::Utf8("a".into())),
            &s
        ),
        Some(Pred::ge(Expr::column(["p"]), Expr::literal("a")))
    );
    assert_eq!(
        to_kernel(&PushdownExpr::IsNull(col("p")), &s),
        Some(Pred::is_null(Expr::column(["p"])))
    );
    let nested = PushdownExpr::eq(
        ColumnPath(vec!["s".into(), "x".into()]),
        ScalarValue::Int32(1),
    );
    assert_eq!(
        to_kernel(&nested, &s),
        Some(Pred::eq(Expr::column(["s", "x"]), Expr::literal(1_i32)))
    );
    let inl = PushdownExpr::In {
        column: col("id"),
        values: vec![ScalarValue::Int64(1), ScalarValue::Int64(2)],
        collation: Collation::Utf8Binary,
    };
    assert_eq!(
        to_kernel(&inl, &s),
        Some(Pred::or_from([
            Pred::eq(Expr::column(["id"]), Expr::literal(1_i64)),
            Pred::eq(Expr::column(["id"]), Expr::literal(2_i64)),
        ]))
    );
    let dec = PushdownExpr::eq(
        col("d"),
        ScalarValue::Decimal128 {
            unscaled: 125,
            precision: 10,
            scale: 2,
        },
    );
    assert!(to_kernel(&dec, &s).is_some());
}

#[test]
fn what_statistics_cannot_answer_stays_in_the_filter() {
    let s = schema();
    // Another type than the column's, a float column, a collation, an unknown column.
    for e in [
        eq("id", ScalarValue::Int32(5)),
        eq("f", ScalarValue::Float64(1.0)),
        eq("nope", ScalarValue::Int64(1)),
        PushdownExpr::Cmp {
            column: col("p"),
            op: CmpOp::Eq,
            value: ScalarValue::Utf8("a".into()),
            collation: Collation::Other("UTF8_LCASE".into()),
        },
        PushdownExpr::Constant(false),
    ] {
        assert_eq!(to_kernel(&e, &s), None, "{e:?}");
    }
}

#[test]
fn and_keeps_what_translates_but_or_and_not_need_all_of_it() {
    let s = schema();
    let good = eq("id", ScalarValue::Int64(1));
    let bad = eq("f", ScalarValue::Float64(1.0));
    let kept = Pred::eq(Expr::column(["id"]), Expr::literal(1_i64));
    assert_eq!(
        to_kernel(&PushdownExpr::And(vec![good.clone(), bad.clone()]), &s),
        Some(Pred::and_from([kept.clone()]))
    );
    assert_eq!(
        to_kernel(&PushdownExpr::Or(vec![good.clone(), bad.clone()]), &s),
        None
    );
    // NOT of a partial AND would prune rows that match: refused.
    let partial = PushdownExpr::Not(Box::new(PushdownExpr::And(vec![good.clone(), bad])));
    assert_eq!(to_kernel(&partial, &s), None);
    assert_eq!(
        to_kernel(&PushdownExpr::Not(Box::new(good)), &s),
        Some(Pred::not(kept))
    );
}

#[test]
fn partition_values_parse_as_the_kernel_does() {
    let null = partition_scalar(None, &DataType::INTEGER).unwrap();
    assert!(null.is_null());
    assert_eq!(
        partition_scalar(Some(&String::new()), &DataType::STRING).unwrap(),
        Scalar::String(String::new())
    );
    assert!(
        partition_scalar(Some(&String::new()), &DataType::INTEGER)
            .unwrap()
            .is_null()
    );
    assert_eq!(
        partition_scalar(Some(&"-1".to_owned()), &DataType::INTEGER).unwrap(),
        Scalar::Integer(-1)
    );
    assert!(partition_scalar(Some(&"x".to_owned()), &DataType::INTEGER).is_err());
}
