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
