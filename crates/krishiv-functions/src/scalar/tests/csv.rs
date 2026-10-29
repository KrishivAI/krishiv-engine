// SPDX-License-Identifier: Apache-2.0
//! `from_csv` and `to_csv` bindings: options, modes, quoting and errors.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, StringArray, StructArray};
use arrow_schema::{DataType, Field, Fields};
use krishiv_common::error::ErrorClass;

use super::{bind_scalar, col, lit};
use super::{owned, strs, text};
use crate::scalar::{BindArg, BindOptions};

/// A one-row `MAP<STRING, STRING>` literal from pairs.
fn map_lit(pairs: &[(&str, &str)]) -> ArrayRef {
    use arrow_array::builder::{MapBuilder, StringBuilder};
    let mut builder = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    for (k, v) in pairs {
        builder.keys().append_value(k);
        builder.values().append_value(v);
    }
    builder.append(true).unwrap();
    Arc::new(builder.finish())
}

/// Binds `from_csv` and returns the function plus its literal args for invoke.
fn bind_from_csv(
    csv: &ArrayRef,
    schema: &str,
    options: Option<&ArrayRef>,
) -> (
    Arc<dyn crate::scalar::ScalarFunction>,
    ArrayRef,
    Option<ArrayRef>,
) {
    let schema_arr: ArrayRef = Arc::new(StringArray::from(vec![Some(schema)]));
    let mut bind_args = vec![
        BindArg::Column {
            ty: csv.data_type().clone(),
            nullable: true,
        },
        BindArg::Literal(Arc::clone(&schema_arr)),
    ];
    let opt_clone = options.map(Arc::clone);
    if let Some(o) = &opt_clone {
        bind_args.push(BindArg::Literal(Arc::clone(o)));
    }
    let f = bind_scalar("from_csv", &bind_args, &BindOptions::new()).unwrap();
    (f, schema_arr, opt_clone)
}

/// The struct rows of `from_csv` as debug strings.
fn struct_debug(a: &ArrayRef) -> Vec<String> {
    let s = a.as_struct();
    let names = s.column_names();
    (0..s.len())
        .map(|i| {
            if s.is_null(i) {
                return "NULL".to_owned();
            }
            let mut parts = Vec::new();
            for (k, name) in names.iter().enumerate() {
                let column = s.column(k);
                if column.is_null(i) {
                    parts.push(format!("{name}:NULL"));
                } else {
                    match column.data_type() {
                        DataType::Int32 => parts.push(format!(
                            "{name}:{}",
                            column
                                .as_primitive::<arrow_array::types::Int32Type>()
                                .value(i)
                        )),
                        DataType::Float32 => parts.push(format!(
                            "{name}:{}",
                            column
                                .as_primitive::<arrow_array::types::Float32Type>()
                                .value(i)
                        )),
                        DataType::Float64 => parts.push(format!(
                            "{name}:{}",
                            column
                                .as_primitive::<arrow_array::types::Float64Type>()
                                .value(i)
                        )),
                        DataType::Utf8 => {
                            parts.push(format!("{name}:{}", column.as_string::<i32>().value(i)));
                        }
                        DataType::Utf8View => {
                            parts.push(format!("{name}:{}", column.as_string_view().value(i)));
                        }
                        DataType::Boolean => {
                            parts.push(format!("{name}:{}", column.as_boolean().value(i)));
                        }
                        _ => parts.push(format!("{name}:?")),
                    }
                }
            }
            parts.join(",")
        })
        .collect()
}

#[test]
fn from_csv_basic_and_nulls() {
    let csv = strs(&[Some("1, 3.14"), Some(""), None]);
    let (f, schema, _) = bind_from_csv(&csv, "a INT, f FLOAT", None);
    assert!(matches!(f.return_type(), DataType::Struct(_)));
    let out = f.invoke(&[col(&csv), lit(&schema)], 3).unwrap();
    assert_eq!(struct_debug(&out), ["a:1,f:3.14", "a:NULL,f:NULL", "NULL"]);
}

#[test]
fn from_csv_wrong_arity_and_types() {
    // The binding (not the unknown-function fallback) reports the bad shape:
    // its message names `from_csv` with the argument types.
    for args in [
        vec![],
        vec![
            DataType::Utf8.into(),
            BindArg::Literal(strs(&[Some("a INT")])),
            BindArg::Literal(strs(&[Some("x")])),
            BindArg::Literal(strs(&[Some("y")])),
        ],
        vec![
            DataType::Int32.into(),
            BindArg::Literal(strs(&[Some("a INT")])),
        ],
    ] {
        let err = bind_scalar("from_csv", &args, &BindOptions::new()).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
        assert!(err.to_string().contains("from_csv("), "{err}");
    }
}

#[test]
fn to_csv_basic_and_nulls() {
    let fields: Fields = vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
    ]
    .into();
    let mut a = arrow_array::builder::Int32Builder::new();
    a.append_value(1);
    a.append_null();
    a.append_value(3);
    let mut b = arrow_array::builder::Int32Builder::new();
    b.append_value(2);
    b.append_null();
    b.append_null();
    let input: ArrayRef = Arc::new(StructArray::new(
        fields,
        vec![Arc::new(a.finish()) as ArrayRef, Arc::new(b.finish())],
        None,
    ));
    let f = bind_scalar(
        "to_csv",
        &[BindArg::Column {
            ty: input.data_type().clone(),
            nullable: true,
        }],
        &BindOptions::new(),
    )
    .unwrap();
    assert_eq!(f.return_type(), &DataType::Utf8);
    let out = f.invoke(&[col(&input)], 3).unwrap();
    assert_eq!(text(&out), owned(&[Some("1,2"), Some(","), Some("3,")]));
}

#[test]
fn to_csv_quoting_and_empty() {
    let fields: Fields = vec![Field::new("v", DataType::Utf8View, true)].into();
    let input: ArrayRef = Arc::new(StructArray::new(
        fields,
        vec![Arc::new(arrow_array::StringViewArray::from(vec![
            Some("a,b"),
            Some("a\"b"),
            Some(""),
            None,
        ])) as ArrayRef],
        None,
    ));
    let f = bind_scalar(
        "to_csv",
        &[BindArg::Column {
            ty: input.data_type().clone(),
            nullable: true,
        }],
        &BindOptions::new(),
    )
    .unwrap();
    let out = f.invoke(&[col(&input)], 4).unwrap();
    assert_eq!(
        text(&out),
        owned(&[Some("\"a,b\""), Some("\"a\\\"b\""), Some(""), Some("")])
    );
}

#[test]
fn from_csv_quoted_fields_with_delimiters_and_newlines() {
    let csv = strs(&[Some("\"a,b\",c"), Some("\"a\nb\",d")]);
    let (f, schema, _) = bind_from_csv(&csv, "x STRING, y STRING", None);
    let out = f.invoke(&[col(&csv), lit(&schema)], 2).unwrap();
    let s = out.as_struct();
    assert_eq!(s.column(0).as_string_view().value(0), "a,b");
    assert_eq!(s.column(1).as_string_view().value(0), "c");
    assert_eq!(s.column(0).as_string_view().value(1), "a\nb");
}

#[test]
fn from_csv_modes() {
    // PERMISSIVE with a corrupt column keeps the raw record.
    let csv = strs(&[Some("1,abc"), Some("notanint,xyz")]);
    let (f, schema, _) = bind_from_csv(&csv, "a INT, _corrupt_record STRING", None);
    let out = f.invoke(&[col(&csv), lit(&schema)], 2).unwrap();
    let s = out.as_struct();
    assert!(
        s.column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .is_valid(0)
    );
    assert!(
        s.column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .is_null(1)
    );
    assert_eq!(s.column(1).as_string_view().value(1), "notanint,xyz");
    // DROPMALFORMED yields NULL rows for malformed records.
    let opts = map_lit(&[("mode", "DROPMALFORMED")]);
    let schema_arr: ArrayRef = Arc::new(StringArray::from(vec![Some("a INT")]));
    let f = bind_scalar(
        "from_csv",
        &[
            BindArg::Column {
                ty: DataType::Utf8,
                nullable: true,
            },
            BindArg::Literal(Arc::clone(&schema_arr)),
            BindArg::Literal(Arc::clone(&opts)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    let csv = strs(&[Some("abc")]);
    let out = f
        .invoke(&[col(&csv), lit(&schema_arr), lit(&opts)], 1)
        .unwrap();
    assert!(out.as_struct().is_null(0) || out.as_struct().column(0).is_null(0));
    // FAILFAST raises.
    let opts = map_lit(&[("mode", "FAILFAST")]);
    let schema_arr: ArrayRef = Arc::new(StringArray::from(vec![Some("a INT")]));
    let f = bind_scalar(
        "from_csv",
        &[
            BindArg::Column {
                ty: DataType::Utf8,
                nullable: true,
            },
            BindArg::Literal(Arc::clone(&schema_arr)),
            BindArg::Literal(Arc::clone(&opts)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    let err = f
        .invoke(&[col(&csv), lit(&schema_arr), lit(&opts)], 1)
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::MalformedRecordInParsingWithoutSuggestion
    );
}

#[test]
fn from_csv_options_sep_and_nullvalue() {
    let csv = strs(&[Some("1|2"), Some("NULL|3")]);
    let opts = map_lit(&[("sep", "|"), ("nullValue", "NULL")]);
    let schema_arr: ArrayRef = Arc::new(StringArray::from(vec![Some("a INT, b INT")]));
    let f = bind_scalar(
        "from_csv",
        &[
            BindArg::Column {
                ty: DataType::Utf8,
                nullable: true,
            },
            BindArg::Literal(Arc::clone(&schema_arr)),
            BindArg::Literal(Arc::clone(&opts)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    let out = f
        .invoke(&[col(&csv), lit(&schema_arr), lit(&opts)], 2)
        .unwrap();
    let s = out.as_struct();
    let a = s.column(0).as_primitive::<arrow_array::types::Int32Type>();
    assert_eq!((a.value(0), a.is_null(1)), (1, true));
}

#[test]
fn from_csv_nan_and_infinities() {
    // Custom spellings of NaN and infinities, as the file format reads them.
    let csv = strs(&[Some("N/A,inf,-inf")]);
    let opts = map_lit(&[
        ("nanValue", "N/A"),
        ("positiveInf", "inf"),
        ("negativeInf", "-inf"),
    ]);
    let (f, schema, _) = bind_from_csv(&csv, "x DOUBLE, y DOUBLE, z DOUBLE", Some(&opts));
    let out = f.invoke(&[col(&csv), lit(&schema), lit(&opts)], 1).unwrap();
    let s = out.as_struct();
    for k in 0..3 {
        let v = s
            .column(k)
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(0);
        assert!(v.is_nan() || v.is_infinite(), "{v}");
    }
    assert!(
        s.column(0)
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(0)
            .is_nan()
    );
    assert_eq!(
        s.column(1)
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(0),
        f64::INFINITY
    );
    assert_eq!(
        s.column(2)
            .as_primitive::<arrow_array::types::Float64Type>()
            .value(0),
        f64::NEG_INFINITY
    );
}

#[test]
fn from_csv_fallback_and_escape_escape() {
    // `enableDateTimeParsingFallback` falls back to the lenient grammar when
    // the strict pattern fails.
    let csv = strs(&[Some("2015-08-26")]);
    let opts = map_lit(&[
        ("dateFormat", "dd/MM/yyyy"),
        ("enableDateTimeParsingFallback", "true"),
    ]);
    let (f, schema, _) = bind_from_csv(&csv, "d DATE", Some(&opts));
    let out = f.invoke(&[col(&csv), lit(&schema), lit(&opts)], 1).unwrap();
    assert!(!out.as_struct().column(0).is_null(0));
    // `charToEscapeQuoteEscaping` escapes the escape character.
    let csv = strs(&[Some("\"a\\\\b\"")]);
    let opts = map_lit(&[("charToEscapeQuoteEscaping", "\\")]);
    let (f, schema, _) = bind_from_csv(&csv, "v STRING", Some(&opts));
    let out = f.invoke(&[col(&csv), lit(&schema), lit(&opts)], 1).unwrap();
    assert_eq!(out.as_struct().column(0).as_string_view().value(0), "a\\b");
}

#[test]
fn round_trip_on_supported_types() {
    let fields: Fields = vec![
        Field::new("i", DataType::Int32, true),
        Field::new("s", DataType::Utf8View, true),
        Field::new("b", DataType::Boolean, true),
    ]
    .into();
    let input: ArrayRef = Arc::new(StructArray::new(
        fields,
        vec![
            Arc::new(arrow_array::Int32Array::from(vec![Some(7), None])) as ArrayRef,
            Arc::new(arrow_array::StringViewArray::from(vec![Some("hi"), None])) as ArrayRef,
            Arc::new(arrow_array::BooleanArray::from(vec![Some(true), None])) as ArrayRef,
        ],
        None,
    ));
    let ser = bind_scalar(
        "to_csv",
        &[BindArg::Column {
            ty: input.data_type().clone(),
            nullable: true,
        }],
        &BindOptions::new(),
    )
    .unwrap();
    let csv = ser.invoke(&[col(&input)], 2).unwrap();
    let (de, schema, _) = bind_from_csv(&csv, "i INT, s STRING, b BOOLEAN", None);
    let back = de.invoke(&[col(&csv), lit(&schema)], 2).unwrap();
    let s = back.as_struct();
    assert_eq!(
        s.column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(0),
        7
    );
    assert!(
        s.column(0)
            .as_primitive::<arrow_array::types::Int32Type>()
            .is_null(1)
    );
}

#[test]
fn rejects_invalid_options_and_nested_types() {
    // An unknown option is unsupported (the message names it, which the
    // unknown-function fallback would not).
    let opts = map_lit(&[("noSuchOption", "x")]);
    let schema_arr: ArrayRef = Arc::new(StringArray::from(vec![Some("a INT")]));
    let err = bind_scalar(
        "from_csv",
        &[
            BindArg::Column {
                ty: DataType::Utf8,
                nullable: true,
            },
            BindArg::Literal(schema_arr),
            BindArg::Literal(opts),
        ],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("noSuchOption"), "{err}");
    // A bad mode is unsupported (analysis would have caught a non-map, but
    // the value check lives at bind).
    let opts = map_lit(&[("mode", "WEIRD")]);
    let schema_arr: ArrayRef = Arc::new(StringArray::from(vec![Some("a INT")]));
    let err = bind_scalar(
        "from_csv",
        &[
            BindArg::Column {
                ty: DataType::Utf8,
                nullable: true,
            },
            BindArg::Literal(schema_arr),
            BindArg::Literal(opts),
        ],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("mode=WEIRD"), "{err}");
    // A nested field type is rejected.
    let csv = strs(&[Some("1")]);
    let schema_arr: ArrayRef = Arc::new(StringArray::from(vec![Some("a ARRAY<INT>")]));
    let err = bind_scalar(
        "from_csv",
        &[
            BindArg::Column {
                ty: csv.data_type().clone(),
                nullable: true,
            },
            BindArg::Literal(schema_arr),
        ],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("nested"), "{err}");
    // `to_csv` over a struct with a nested field raises the datasource error.
    let fields: Fields = vec![Field::new(
        "a",
        DataType::List(Arc::new(Field::new("element", DataType::Int32, true))),
        true,
    )]
    .into();
    let input: ArrayRef = Arc::new(StructArray::new(
        fields,
        vec![arrow_array::new_null_array(
            &DataType::List(Arc::new(Field::new("element", DataType::Int32, true))),
            1,
        )],
        None,
    ));
    let err = bind_scalar(
        "to_csv",
        &[BindArg::Column {
            ty: input.data_type().clone(),
            nullable: true,
        }],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::UnsupportedDataTypeForDatasource
    );
    assert!(err.to_string().contains("CSV"), "{err}");
}

#[test]
fn empty_input_and_type_failures() {
    // Empty input parses to one empty field (then malformed for a 2-field schema).
    let csv = strs(&[Some("")]);
    let (f, schema, _) = bind_from_csv(&csv, "a INT, b INT", None);
    let out = f.invoke(&[col(&csv), lit(&schema)], 1).unwrap();
    assert!(out.as_struct().column(0).is_null(0));
    // A float that does not parse is malformed (PERMISSIVE: null).
    let csv = strs(&[Some("abc")]);
    let (f, schema, _) = bind_from_csv(&csv, "f FLOAT", None);
    let out = f.invoke(&[col(&csv), lit(&schema)], 1).unwrap();
    assert!(out.as_struct().column(0).is_null(0));
}
