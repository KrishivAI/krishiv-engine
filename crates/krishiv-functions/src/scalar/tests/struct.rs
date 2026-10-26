// SPDX-License-Identifier: Apache-2.0
//! The struct bindings: `named_struct` and `struct` pack their arguments.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array, StringArray};
use arrow_schema::{DataType, Field};
use krishiv_common::error::ErrorClass;

use super::{Arg, BindArg, BindOptions, bind_scalar, col, lit};

fn ints(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(v.to_vec()))
}

fn strs(v: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(v.to_vec()))
}

fn one(v: &ArrayRef) -> ArrayRef {
    arrow_array::ArrayRef::from(v.slice(0, 1))
}

/// Binds `name` over literal `names` and columns `values`.
fn bind_named(names: &[ArrayRef], values: &[ArrayRef]) -> Arc<dyn super::ScalarFunction> {
    let mut args = Vec::new();
    for (n, v) in names.iter().zip(values) {
        args.push(BindArg::Literal(Arc::clone(n)));
        args.push(BindArg::Column {
            ty: v.data_type().clone(),
            nullable: true,
        });
    }
    bind_scalar("named_struct", &args, &BindOptions::new()).unwrap()
}

#[test]
fn named_struct_packs_values_under_folded_names() {
    let values = [
        ints(&[Some(1), None, Some(3)]),
        strs(&[Some("a"), None, None]),
    ];
    let f = bind_named(&[strs(&[Some("x")]), strs(&[Some("y")])], &values);
    assert_eq!(
        f.return_type(),
        &DataType::Struct(
            vec![
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Utf8, true),
            ]
            .into()
        )
    );
    let out = f
        .invoke(
            &[
                lit(&one(&strs(&[Some("x")]))),
                col(&values[0]),
                lit(&one(&strs(&[Some("y")]))),
                col(&values[1]),
            ],
            3,
        )
        .unwrap();
    let s = out.as_struct();
    assert_eq!(s.len(), 3);
    assert_eq!(s.column_names(), &["x", "y"]);
    assert!(s.column(0).is_null(1));
    assert!(s.column(1).is_null(1));
    assert!(!s.column(0).is_null(2));
}

#[test]
fn struct_names_fields_positionally() {
    let values = [ints(&[Some(1), None]), strs(&[Some("a"), None])];
    let args: Vec<BindArg> = values
        .iter()
        .map(|v| BindArg::Column {
            ty: v.data_type().clone(),
            nullable: true,
        })
        .collect();
    let f = bind_scalar("struct", &args, &BindOptions::new()).unwrap();
    assert_eq!(
        f.return_type(),
        &DataType::Struct(
            vec![
                Field::new("col1", DataType::Int32, true),
                Field::new("col2", DataType::Utf8, true),
            ]
            .into()
        )
    );
    let out = f.invoke(&[col(&values[0]), col(&values[1])], 2).unwrap();
    let s = out.as_struct();
    assert_eq!(s.column_names(), &["col1", "col2"]);
    assert!(s.column(1).is_null(1));
}

#[test]
fn struct_field_nullability_follows_the_arguments() {
    let non_null = || BindArg::Column {
        ty: DataType::Int32,
        nullable: false,
    };
    let f = bind_scalar(
        "struct",
        &[non_null(), DataType::Int32.into()],
        &BindOptions::new(),
    )
    .unwrap();
    let DataType::Struct(fields) = f.return_type() else {
        panic!("a struct return type");
    };
    assert!(!fields[0].is_nullable());
    assert!(fields[1].is_nullable());
}

#[test]
fn named_struct_needs_literal_names_and_even_arity() {
    let lit = || BindArg::Literal(strs(&[Some("a")]));
    let col = || BindArg::Column {
        ty: DataType::Int32,
        nullable: true,
    };
    // Odd arity.
    let err = bind_scalar("named_struct", &[lit(), col(), lit()], &BindOptions::new()).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // A column name is not a literal.
    let err = bind_scalar("named_struct", &[col(), col()], &BindOptions::new()).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // A null name is not a name.
    let err = bind_scalar(
        "named_struct",
        &[BindArg::Literal(strs(&[None])), col()],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // No arguments at all.
    let err = bind_scalar("struct", &[], &BindOptions::new()).unwrap();
    assert_eq!(
        err.return_type(),
        &DataType::Struct(Vec::<Field>::new().into())
    );
    let _ = Arg::Array(ints(&[Some(1)]));
}
