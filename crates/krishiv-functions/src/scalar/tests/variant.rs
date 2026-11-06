// SPDX-License-Identifier: Apache-2.0
//! The variant function bindings (Spark 4.2.0 `variantExpressions.scala`).

use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::{Array, ArrayRef, StringArray};
use krishiv_common::error::ErrorClass;

use super::{Arg, BindArg, BindOptions, bind_scalar};

fn lit(s: Option<&str>) -> BindArg {
    BindArg::Literal(Arc::new(StringArray::from(vec![s])))
}

fn variants(json: &[Option<&str>]) -> ArrayRef {
    let f = bind_scalar(
        "parse_json",
        &[BindArg::Column {
            ty: arrow_schema::DataType::Utf8,
            nullable: true,
        }],
        &BindOptions::default(),
    )
    .unwrap();
    let input: ArrayRef = Arc::new(StringArray::from(json.to_vec()));
    f.invoke(&[Arg::Array(input)], json.len()).unwrap()
}

#[test]
fn a_bad_literal_path_fails_at_bind() {
    let v = BindArg::Column {
        ty: krishiv_types::DataType::Variant.to_arrow(),
        nullable: true,
    };
    let e = bind_scalar(
        "variant_get",
        &[v, lit(Some("$[x]"))],
        &BindOptions::default(),
    )
    .err()
    .unwrap();
    assert_eq!(e.error_class(), ErrorClass::InvalidVariantGetPath);
}

#[test]
fn variant_get_reads_a_column_of_paths() {
    let col = variants(&[Some(r#"{"a":1,"b":"x"}"#), Some(r#"{"a":2}"#), Some("[3]")]);
    let v = BindArg::Column {
        ty: col.data_type().clone(),
        nullable: true,
    };
    let p = BindArg::Column {
        ty: arrow_schema::DataType::Utf8,
        nullable: true,
    };
    let f = bind_scalar(
        "try_variant_get",
        &[v, p, lit(Some("int"))],
        &BindOptions::default(),
    )
    .unwrap();
    let paths: ArrayRef = Arc::new(StringArray::from(vec![Some("$.b"), None, Some("$[0]")]));
    let int: ArrayRef = Arc::new(StringArray::from(vec!["int"]));
    let out = f
        .invoke(
            &[
                Arg::Array(Arc::clone(&col)),
                Arg::Array(paths),
                Arg::Scalar(Arc::clone(&int)),
            ],
            3,
        )
        .unwrap();
    let out = out.as_primitive::<arrow_array::types::Int32Type>();
    // `"x"` is no int (NULL under try), a NULL path is NULL, `[3][0]` is 3.
    assert!(out.is_null(0) && out.is_null(1));
    assert_eq!(out.value(2), 3);
    // A bad path in a column still raises, even for try_variant_get.
    let bad: ArrayRef = Arc::new(StringArray::from(vec![Some("$.a"), Some("$["), None]));
    let e = f
        .invoke(&[Arg::Array(col), Arg::Array(bad), Arg::Scalar(int)], 3)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidVariantGetPath);
}

fn column_arg(a: &ArrayRef) -> BindArg {
    BindArg::Column {
        ty: a.data_type().clone(),
        nullable: true,
    }
}

#[test]
fn the_variant_functions_bind_and_run() {
    let opts = BindOptions::default();
    let col = variants(&[Some(r#"{"a":null}"#), Some("null"), None]);
    // to_json of a variant prints it; is_variant_null sees only variant nulls.
    let f = bind_scalar("to_json", &[column_arg(&col)], &opts).unwrap();
    let out = f.invoke(&[Arg::Array(Arc::clone(&col))], 3).unwrap();
    let out: Vec<_> = out.as_string_view().iter().collect();
    assert_eq!(out, vec![Some(r#"{"a":null}"#), Some("null"), None]);
    let f = bind_scalar("is_variant_null", &[column_arg(&col)], &opts).unwrap();
    let out = f.invoke(&[Arg::Array(Arc::clone(&col))], 3).unwrap();
    let out: Vec<_> = out.as_boolean().iter().collect();
    assert_eq!(out, vec![Some(false), Some(true), Some(false)]);
    // try_parse_json is NULL where parse_json raises.
    let text: ArrayRef = Arc::new(StringArray::from(vec![Some("{"), Some("1")]));
    let t = BindArg::Column {
        ty: arrow_schema::DataType::Utf8,
        nullable: true,
    };
    let f = bind_scalar("try_parse_json", std::slice::from_ref(&t), &opts).unwrap();
    let out = f.invoke(&[Arg::Array(Arc::clone(&text))], 2).unwrap();
    assert!(out.is_null(0) && out.is_valid(1));
    let f = bind_scalar("parse_json", &[t], &opts).unwrap();
    assert!(f.invoke(&[Arg::Array(text)], 2).is_err());
    // variant_get with a NULL literal path is NULL; without a type, a variant.
    let v = column_arg(&col);
    let f = bind_scalar("variant_get", &[v.clone(), lit(None)], &opts).unwrap();
    let null_path: ArrayRef = Arc::new(StringArray::from(vec![None::<&str>]));
    let out = f
        .invoke(&[Arg::Array(Arc::clone(&col)), Arg::Scalar(null_path)], 3)
        .unwrap();
    assert_eq!(out.null_count(), 3);
    let f = bind_scalar("variant_get", &[v, lit(Some("$.a"))], &opts).unwrap();
    let path: ArrayRef = Arc::new(StringArray::from(vec!["$.a"]));
    let out = f.invoke(&[Arg::Array(col), Arg::Scalar(path)], 3).unwrap();
    // `{"a":null}`'s `a` is a variant null (a value); the others miss.
    assert!(out.is_valid(0) && out.is_null(1) && out.is_null(2));
    // A wrong argument type does not bind.
    assert!(bind_scalar("is_variant_null", &[lit(Some("x"))], &opts).is_err());
}
