// SPDX-License-Identifier: Apache-2.0
//! Signature tests, part of the registry tests.

use super::*;

#[test]
fn map_signatures() {
    use krishiv_types::{ArrayType, MapType, StructField, StructType};
    let (i, s) = (ty(DataType::Integer), ty(DataType::String));
    let map =
        |k: DataType, v: DataType, vcn: bool| ArgType::new(MapType::new(k, v, vcn).into(), false);
    assert_eq!(def("map").inputs, Inputs::Map);
    assert_eq!(def("map_concat").inputs, Inputs::MapConcat);
    assert_eq!(def("map_contains_key").inputs, Inputs::MapContainsKey);
    // `map(k, v)`: the key and value types, never null; values may hold
    // null when any value argument may.
    assert_eq!(
        result("map", &[i.clone(), s.clone(), i.clone(), s.clone()], &[]).unwrap(),
        map(DataType::Integer, DataType::String, false)
    );
    assert_eq!(
        result(
            "map",
            &[i.clone(), ArgType::new(DataType::String, true)],
            &[]
        )
        .unwrap(),
        ArgType::new(
            MapType::new(DataType::Integer, DataType::String, true).into(),
            false
        )
    );
    // A key holding a map is `INVALID_MAP_KEY_TYPE`.
    let nested: DataType = MapType::new(DataType::String, DataType::Integer, true).into();
    let e = result("map", &[ty(nested.clone()), s.clone()], &[]).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchInvalidMapKeyType
    );
    for name in ["map_concat", "map_from_arrays"] {
        let key = if name == "map_concat" {
            MapType::new(nested.clone(), DataType::String, true).into()
        } else {
            ArrayType::new(nested.clone(), false).into()
        };
        let value = if name == "map_concat" {
            nested.clone()
        } else {
            ArrayType::new(DataType::String, false).into()
        };
        let e = result(name, &[ty(key), ty(value)], &[]).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::DatatypeMismatchInvalidMapKeyType,
            "{name}"
        );
    }
    // `map_concat(m)`: the merged map, null when any argument may be.
    assert_eq!(
        result(
            "map_concat",
            &[
                ArgType::new(
                    MapType::new(DataType::Integer, DataType::String, false).into(),
                    true
                ),
                ArgType::new(
                    MapType::new(DataType::Integer, DataType::String, true).into(),
                    false
                ),
            ],
            &[]
        )
        .unwrap(),
        ArgType::new(
            MapType::new(DataType::Integer, DataType::String, true).into(),
            true
        )
    );
    // `map_contains_key(map, key)`: a nullable boolean, `MAP_FUNCTION_DIFF_TYPES`
    // for a key of another type, `NULL_TYPE` for an untyped null key.
    assert_eq!(
        result(
            "map_contains_key",
            &[map(DataType::Integer, DataType::String, false), i.clone()],
            &[]
        )
        .unwrap(),
        ArgType::new(DataType::Boolean, true)
    );
    let e = result(
        "map_contains_key",
        &[map(DataType::Integer, DataType::String, false), s.clone()],
        &[],
    )
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchMapFunctionDiffTypes
    );
    let e = result(
        "map_contains_key",
        &[map(DataType::Integer, DataType::String, false), null_ty()],
        &[],
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DatatypeMismatchNullType);
    // `map_keys`, `map_values`, `map_entries`: the projections.
    let m = ArgType::new(
        MapType::new(DataType::Integer, DataType::String, true).into(),
        true,
    );
    assert_eq!(
        result("map_keys", std::slice::from_ref(&m), &[]).unwrap(),
        ArgType::new(ArrayType::new(DataType::Integer, true).into(), true)
    );
    assert_eq!(
        result("map_values", std::slice::from_ref(&m), &[]).unwrap(),
        ArgType::new(ArrayType::new(DataType::String, true).into(), true)
    );
    assert_eq!(
        result("map_entries", std::slice::from_ref(&m), &[]).unwrap(),
        ArgType::new(
            ArrayType::new(
                StructType::new([
                    StructField::new("key", DataType::Integer, false),
                    StructField::new("value", DataType::String, true),
                ])
                .into(),
                false
            )
            .into(),
            true
        )
    );
    let e = result("map_keys", std::slice::from_ref(&i), &[]).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    // `map_from_arrays(keys, values)`: the map of the element types.
    assert_eq!(
        result(
            "map_from_arrays",
            &[
                ty(ArrayType::new(DataType::Integer, false).into()),
                ty(ArrayType::new(DataType::String, false).into()),
            ],
            &[]
        )
        .unwrap(),
        map(DataType::Integer, DataType::String, false)
    );
    // `map_from_entries`: the map of the struct fields; anything else is
    // `UNEXPECTED_INPUT_TYPE`.
    let pair = ArrayType::new(
        StructType::new([
            StructField::new("key", DataType::Integer, true),
            StructField::new("value", DataType::String, true),
        ])
        .into(),
        true,
    )
    .into();
    assert_eq!(
        result("map_from_entries", &[ty(pair)], &[]).unwrap(),
        ArgType::new(
            MapType::new(DataType::Integer, DataType::String, true).into(),
            true
        )
    );
    let e = result("map_from_entries", std::slice::from_ref(&i), &[]).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    // `str_to_map(text)`: `map<string, string>`.
    assert_eq!(
        result("str_to_map", std::slice::from_ref(&s), &[]).unwrap(),
        ArgType::new(
            MapType::new(DataType::String, DataType::String, true).into(),
            false
        )
    );
}

#[test]
fn hof_signatures() {
    use DataType::{Array, Boolean, Integer};
    let array = |e: DataType, contains_null: bool, nullable: bool| {
        ArgType::new(Array(ArrayType::new(e, contains_null).into()), nullable)
    };
    let i = || ty(Integer);
    let b = || ty(Boolean);
    // `array(...)`: the common element type; untyped `NULL` is `VOID`,
    // no arguments an empty `VOID` array.
    assert_eq!(
        result("array", &[i(), i()], &[]).unwrap(),
        array(Integer, false, false)
    );
    assert_eq!(
        result("array", &[ArgType::new(Integer, true), i()], &[]).unwrap(),
        array(Integer, true, false)
    );
    assert_eq!(
        result("array", &[null_ty(), null_ty()], &[]).unwrap(),
        array(DataType::Null, true, false)
    );
    assert_eq!(
        result("array", &[], &[]).unwrap(),
        array(DataType::Null, false, false)
    );
    // The higher-order functions type their lambda argument as its body:
    // `transform` maps the body, `filter` keeps the array, `exists` and
    // `forall` fold to booleans.
    let arr = || array(Integer, true, false);
    assert_eq!(
        result("transform", &[arr(), i()], &[]).unwrap(),
        array(Integer, false, false)
    );
    assert_eq!(
        result("transform", &[arr(), ArgType::new(Integer, true)], &[]).unwrap(),
        array(Integer, true, false)
    );
    assert_eq!(result("filter", &[arr(), b()], &[]).unwrap(), arr());
    assert_eq!(
        result("exists", &[array(Integer, false, true), b()], &[]).unwrap(),
        ArgType::new(Boolean, true)
    );
    // `exists`/`forall` are nullable when the array is or when the body
    // is (Spark derives the nullability from the array and the body
    // alone): only a non-null array with a non-null body folds to a
    // non-null boolean.
    assert_eq!(
        result("forall", &[arr(), b()], &[]).unwrap(),
        ArgType::new(Boolean, false)
    );
    assert_eq!(
        result(
            "exists",
            &[array(Integer, false, false), ArgType::new(Boolean, true)],
            &[]
        )
        .unwrap(),
        ArgType::new(Boolean, true)
    );
    assert_eq!(
        result("forall", &[array(Integer, false, false), b()], &[]).unwrap(),
        ArgType::new(Boolean, false)
    );
    // The array comes first and holds an array; the predicate is boolean.
    for name in ["transform", "filter", "exists", "forall"] {
        let err = result(name, &[i(), i()], &[]).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{name}"
        );
        assert!(err.to_string().contains("\"ARRAY\""), "{name}: {err}");
    }
    for name in ["filter", "exists", "forall"] {
        let err = result(name, &[arr(), i()], &[]).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{name}"
        );
        assert!(err.to_string().contains("\"BOOLEAN\""), "{name}: {err}");
    }
    // Higher-order calls run in the executor, never in a kernel binding.
    for name in [
        "transform",
        "filter",
        "exists",
        "forall",
        "aggregate",
        "reduce",
        "zip_with",
        "map_filter",
        "map_zip_with",
        "transform_keys",
        "transform_values",
        "array_sort",
    ] {
        assert_eq!(def(name).execution, Execution::HigherOrder, "{name}");
        assert_eq!(def(name).kind, FunctionKind::Scalar, "{name}");
    }
    assert_eq!(def("array").execution, Execution::Scalar);
    assert_eq!(def("map").execution, Execution::Scalar);
    assert_eq!(def("size").execution, Execution::Scalar);
    assert_eq!(def("split").execution, Execution::Scalar);
    // The arity check runs before typing, so a call typed without its
    // array argument is an engine bug, never a user error.
    for name in ["transform", "filter", "exists", "forall"] {
        let err = result(name, &[], &[]).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::InternalError, "{name}");
    }
}

#[test]
fn aggregate_types_its_fold() {
    use DataType::{Array as Arr, Boolean, Integer};
    let array = |e: DataType, contains_null: bool, nullable: bool| {
        ArgType::new(Arr(ArrayType::new(e, contains_null).into()), nullable)
    };
    let arr = || array(Integer, true, false);
    let i = || ArgType::new(Integer, false);
    // The merge body's type is the accumulator's (the zero's) type; the
    // result is the finish body's, or the accumulator's without one.
    assert_eq!(
        result("aggregate", &[arr(), i(), i()], &[]).unwrap(),
        ArgType::new(Integer, true)
    );
    assert_eq!(
        result("reduce", &[arr(), i(), i()], &[]).unwrap(),
        ArgType::new(Integer, true)
    );
    // A merge of another type is `UNEXPECTED_INPUT_TYPE` on the third
    // argument, quoting the accumulator's type.
    for name in ["aggregate", "reduce"] {
        let err = result(name, &[arr(), i(), ty(Boolean)], &[]).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{name}"
        );
        assert!(err.to_string().contains("\"INT\""), "{name}: {err}");
    }
    // The finish body's type is the result's.
    assert_eq!(
        result("aggregate", &[arr(), i(), i(), ty(Boolean)], &[])
            .unwrap()
            .data_type,
        Some(Boolean)
    );
}

#[test]
fn zip_with_types_its_merged_array() {
    use DataType::{Array as Arr, Integer, String as Text};
    let array = |e: DataType| ArgType::new(Arr(ArrayType::new(e, true).into()), false);
    let r = result("zip_with", &[array(Integer), array(Text), ty(Text)], &[]).unwrap();
    assert_eq!(
        r,
        ArgType::new(Arr(ArrayType::new(Text, false).into()), false)
    );
    // Both sides hold arrays.
    for bad in [
        result("zip_with", &[ty(Integer), array(Text), ty(Text)], &[]),
        result("zip_with", &[array(Integer), ty(Integer), ty(Text)], &[]),
    ] {
        assert_eq!(
            bad.unwrap_err().error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType
        );
    }
}

#[test]
fn map_functions_type_their_maps() {
    use DataType::{Boolean, Integer, Map as MapTy, String as Text};
    use krishiv_types::MapType;
    let map = |k: DataType, v: DataType, nullable: bool| {
        ArgType::new(MapTy(MapType::new(k, v, true).into()), nullable)
    };
    let m = || map(Text, Integer, false);
    // `map_filter` keeps the map, with a boolean body.
    assert_eq!(result("map_filter", &[m(), ty(Boolean)], &[]).unwrap(), m());
    let err = result("map_filter", &[m(), ty(Integer)], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    // `transform_keys` maps the key type, refusing map keys at analysis.
    let r = result("transform_keys", &[m(), ty(Integer)], &[]).unwrap();
    assert_eq!(
        r.data_type,
        Some(MapTy(MapType::new(Integer, Integer, true).into()))
    );
    let err = result(
        "transform_keys",
        &[
            m(),
            ArgType::new(MapTy(MapType::new(Text, Text, true).into()), false),
        ],
        &[],
    )
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchInvalidMapKeyType
    );
    // `transform_values` maps the value type.
    let r = result("transform_values", &[m(), ty(Text)], &[]).unwrap();
    assert_eq!(
        r.data_type,
        Some(MapTy(MapType::new(Text, Text, false).into()))
    );
    // `map_zip_with` unions the key sets; the keys must match.
    let r = result("map_zip_with", &[m(), m(), ty(Integer)], &[]).unwrap();
    assert_eq!(
        r.data_type,
        Some(MapTy(MapType::new(Text, Integer, false).into()))
    );
    let err = result(
        "map_zip_with",
        &[m(), map(Integer, Integer, false), ty(Integer)],
        &[],
    )
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchMapZipWithDiffTypes
    );
    // The map comes first and holds a map.
    let err = result("map_filter", &[ty(Integer), ty(Boolean)], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
}

#[test]
fn array_sort_types_its_array() {
    use DataType::{Array as Arr, Integer};
    let array = || ArgType::new(Arr(ArrayType::new(Integer, true).into()), false);
    // Without a comparator the array sorts as it is.
    assert_eq!(result("array_sort", &[array()], &[]).unwrap(), array());
    // The comparator body is `INT`.
    assert_eq!(
        result("array_sort", &[array(), ty(Integer)], &[]).unwrap(),
        array()
    );
    let err = result("array_sort", &[array(), ty(DataType::Boolean)], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedReturnType
    );
}

#[test]
fn size_and_split_type_their_collections() {
    use DataType::{Array as Arr, Integer, String as Text};
    // `size` counts arrays and maps.
    assert_eq!(
        result(
            "size",
            &[ArgType::new(
                Arr(ArrayType::new(Integer, true).into()),
                true
            )],
            &[]
        )
        .unwrap()
        .data_type,
        Some(Integer)
    );
    let err = result("size", &[ty(Integer)], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    // `split` pieces a string around a pattern.
    assert_eq!(
        result("split", &[ty(Text), ty(Text), ty(Integer)], &[])
            .unwrap()
            .data_type,
        Some(Arr(ArrayType::new(Text, false).into()))
    );
}

#[test]
fn moment_aggregates_are_registered_as_doubles() {
    let r = builtins();
    for (name, agg, arguments) in [
        ("stddev", Aggregate::StddevSamp, 1),
        ("std", Aggregate::StddevSamp, 1),
        ("stddev_samp", Aggregate::StddevSamp, 1),
        ("stddev_pop", Aggregate::StddevPop, 1),
        ("variance", Aggregate::VarSamp, 1),
        ("var_samp", Aggregate::VarSamp, 1),
        ("var_pop", Aggregate::VarPop, 1),
        ("skewness", Aggregate::Skewness, 1),
        ("kurtosis", Aggregate::Kurtosis, 1),
        ("corr", Aggregate::Corr, 2),
        ("covar_samp", Aggregate::CovarSamp, 2),
        ("covar_pop", Aggregate::CovarPop, 2),
    ] {
        let d = def(name);
        assert_eq!(d.kind, FunctionKind::Aggregate, "{name}");
        assert_eq!(d.execution, Execution::Aggregate(agg), "{name}");
        assert!(d.deterministic, "{name}");
        assert!(
            d.arity.accepts(arguments) && !d.arity.accepts(3 - arguments),
            "{name}"
        );
        // Found in every profile that has the name, and only there.
        let since = r.spark_function(name).unwrap().since;
        for p in SparkProfile::ALL {
            assert_eq!(
                matches!(r.lookup(name, p), Lookup::Found(_)),
                p >= since,
                "{name} {p:?}"
            );
        }
        // Every argument is cast to DOUBLE: an integer, a decimal and a
        // string become one; the result is a nullable DOUBLE whatever the
        // arguments (no rows, one row).
        let args: Vec<ArgType> = (0..arguments).map(|_| ty(DataType::Integer)).collect();
        assert_eq!(
            d.inputs.expected(&CallTypes::new(&args, &[])),
            Some(vec![InputType::Exact(DataType::Double)]),
            "{name}"
        );
        assert_eq!(
            result(name, &args, &[]).unwrap(),
            ArgType::new(DataType::Double, true),
            "{name}"
        );
    }
}

#[test]
fn moment_aggregates_reject_the_wrong_number_of_arguments() {
    for (name, wrong) in [
        ("stddev", 2),
        ("corr", 1),
        ("covar_pop", 3),
        ("kurtosis", 0),
    ] {
        assert!(!def(name).arity.accepts(wrong), "{name}({wrong})");
    }
}

/// `parse_url` and `try_parse_url`: arities, inputs and result types.
#[test]
fn url_signatures() {
    let s = ty(DataType::String);
    let strict = InputType::Strict(DataType::String);
    // `parse_url`/`try_parse_url` take two or three strict strings (an
    // `int` URL is refused, not cast) and are always nullable.
    for name in ["parse_url", "try_parse_url"] {
        let d = def(name);
        assert!(d.deterministic, "{name}");
        assert!(!d.arity.accepts(1) && d.arity.accepts(2) && d.arity.accepts(3));
        assert!(!d.arity.accepts(4));
        assert_eq!(
            d.inputs,
            Inputs::Expects(vec![strict.clone(), strict.clone(), strict.clone()])
        );
        assert_eq!(
            result(name, &[s.clone(), s.clone()], &[]).unwrap(),
            ArgType::new(DataType::String, true)
        );
        assert_eq!(
            result(name, &[s.clone(), s.clone(), s.clone()], &[]).unwrap(),
            ArgType::new(DataType::String, true)
        );
    }
    // `try_parse_url` arrived in 4.0.
    assert!(matches!(
        builtins().lookup("try_parse_url", SparkProfile::V3_5),
        Lookup::NotFound
    ));
}

/// `schema_of_csv`: Spark's checks in Spark's order (the options map,
/// whether the CSV folds, whether it is null, whether it is a string).
#[test]
fn csv_signatures() {
    let s = ty(DataType::String);
    let d = def("schema_of_csv");
    assert!(d.deterministic);
    assert!(d.arity.accepts(1) && d.arity.accepts(2) && !d.arity.accepts(0));
    assert!(!d.arity.accepts(3));
    assert_eq!(d.inputs, Inputs::Any);
    // A folded string resolves to a non-nullable string (the DDL).
    let csv = |v: &str| [Some(Constant::String(v.to_owned()))];
    assert_eq!(
        result("schema_of_csv", std::slice::from_ref(&s), &csv("1,abc")).unwrap(),
        ty(DataType::String)
    );
    // Anything but a one-argument string map is refused first.
    let err = result("schema_of_csv", &[s.clone(), s.clone()], &csv("1")).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidOptionsNonMapFunction);
    // A non-foldable CSV names itself.
    let err = result("schema_of_csv", std::slice::from_ref(&s), &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    assert!(err.to_string().contains("`csv`"), "{err}");
    // A null CSV is `UNEXPECTED_NULL`.
    let err = result(
        "schema_of_csv",
        std::slice::from_ref(&s),
        &[Some(Constant::Null)],
    )
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedNull
    );
    assert!(err.to_string().contains("must not be null"), "{err}");
    // A folded non-string is the wrong type, not a cast.
    let err = result(
        "schema_of_csv",
        &[ty(DataType::Integer)],
        &[Some(Constant::Integral(42))],
    )
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(err.to_string().contains("\"STRING\""), "{err}");
    // A plain literal the folder has no constant for (a timestamp or a
    // binary) is still folded in Spark: non-null, so the wrong type.
    for t in [DataType::Timestamp, DataType::TimestampNtz, DataType::Date] {
        let args = [ty(t)];
        let err =
            (def("schema_of_csv").result)(&CallTypes::new(&args, &[None]).with_literals(&[true]))
                .unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType
        );
    }
    // A folding failure propagates.
    let failed = Constant::Error(KrishivError::from(
        krishiv_common::error::CastInvalidInput {
            expression: "'abc'".to_owned(),
            source_type: "\"STRING\"".to_owned(),
            target_type: "\"INT\"".to_owned(),
        },
    ));
    let err = result("schema_of_csv", std::slice::from_ref(&s), &[Some(failed)]).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::CastInvalidInput);
}

/// `from_csv` and `to_csv`: Spark's checks (options map, schema shape, first
/// argument types) with the golden's classes and parameters.
#[test]
fn from_to_csv_signatures() {
    use krishiv_types::{MapType, StructField, StructType};
    let s = ty(DataType::String);
    let d = def("from_csv");
    assert!(d.deterministic);
    assert!(d.arity.accepts(2) && d.arity.accepts(3));
    assert!(!d.arity.accepts(1) && !d.arity.accepts(4));
    assert!(def("to_csv").arity.accepts(1) && def("to_csv").arity.accepts(2));
    assert!(!def("to_csv").arity.accepts(0) && !def("to_csv").arity.accepts(3));
    // `from_csv('1, 3.14', 'a INT, f FLOAT')` resolves to its struct.
    let args = [s.clone(), s.clone()];
    let consts = [
        Some(Constant::String("1".to_owned())),
        Some(Constant::String("a INT".to_owned())),
    ];
    let got = result("from_csv", &args, &consts).unwrap();
    assert_eq!(
        got.data_type,
        Some(DataType::Struct(krishiv_types::StructType::new(vec![
            StructField::new("a".to_owned(), DataType::Integer, true),
        ])))
    );
    // A non-string schema is not a schema.
    let err = result(
        "from_csv",
        &[s.clone(), ty(DataType::Integer)],
        &[
            Some(Constant::String("1".to_owned())),
            Some(Constant::Integral(1)),
        ],
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidSchemaNonStringLiteral);
    // A valid type that is not a struct.
    let err = result(
        "from_csv",
        &[s.clone(), s.clone()],
        &[
            Some(Constant::String("1".to_owned())),
            Some(Constant::String("Array<int>".to_owned())),
        ],
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidSchemaNonStructType);
    assert!(err.to_string().contains("ARRAY<INT>"), "{err}");
    // A malformed DDL.
    let err = result(
        "from_csv",
        &[s.clone(), s.clone()],
        &[
            Some(Constant::String("1".to_owned())),
            Some(Constant::String("a InvalidType".to_owned())),
        ],
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::ParseSyntaxError);
    // A non-foldable, non-literal schema is `INVALID_SCHEMA.NON_STRING_LITERAL`
    // (Spark 4.2.0, checked against a live server).
    let err = (def("from_csv").result)(
        &CallTypes::new(
            &[s.clone(), s.clone()],
            &[Some(Constant::String("1".to_owned())), None],
        )
        .with_literals(&[true, false]),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidSchemaNonStringLiteral);
    // Options must be a map.
    let err = result(
        "from_csv",
        &[s.clone(), s.clone(), ty(DataType::String)],
        &[
            Some(Constant::String("1".to_owned())),
            Some(Constant::String("a INT".to_owned())),
            Some(Constant::String("x".to_owned())),
        ],
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidOptionsNonMapFunction);
    // A map with non-string values names its type.
    let map_ty = DataType::Map(Arc::new(MapType::new(
        DataType::String,
        DataType::Integer,
        true,
    )));
    let err = result(
        "from_csv",
        &[s.clone(), s, ArgType::new(map_ty, true)],
        &[
            Some(Constant::String("1".to_owned())),
            Some(Constant::String("a INT".to_owned())),
            None,
        ],
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidOptionsNonStringType);
    assert!(err.to_string().contains("MAP<STRING, INT>"), "{err}");
    // `to_csv` needs a struct.
    let err = result(
        "to_csv",
        &[ty(DataType::Integer)],
        &[Some(Constant::Integral(1))],
    )
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnsupportedInputType
    );
    assert!(err.to_string().contains("`to_csv`"), "{err}");
    let got = result(
        "to_csv",
        &[ArgType::new(
            DataType::Struct(StructType::new(vec![StructField::new(
                "a".to_owned(),
                DataType::Integer,
                true,
            )])),
            true,
        )],
        &[None],
    )
    .unwrap();
    assert_eq!(got.data_type, Some(DataType::String));
}

/// `schema_of_csv` never executes: the analyzer replaces every
/// well-typed call with the DDL literal, so it has no scalar binding. A
/// call that kept its shape (unreachable with the built-in registry: the
/// signature rejects everything the analyzer does not resolve) fails at
/// execution binding rather than running something unassigned.
#[test]
fn schema_of_csv_has_no_scalar_binding() {
    use crate::scalar::{BindArg, BindOptions, bind_scalar};
    assert_eq!(def("schema_of_csv").execution, Execution::Evaluator);
    let args = [BindArg::from(arrow_schema::DataType::Utf8)];
    let err = bind_scalar("schema_of_csv", &args, &BindOptions::new()).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

/// The cast-named functions (`float(x)`, `int(x)`, ...): kind scalar,
/// arity exactly 1 from Spark's catalog, any input, result the target
/// type (`decimal(10,0)` for `decimal`). The analyzer rewrites every
/// call to a cast, so they run as `Execution::Cast` and are never
/// executed or bound as functions.
#[test]
fn cast_named_functions_rewrite_to_casts() {
    let targets = [
        ("boolean", DataType::Boolean),
        ("tinyint", DataType::Byte),
        ("smallint", DataType::Short),
        ("int", DataType::Integer),
        ("bigint", DataType::Long),
        ("float", DataType::Float),
        ("double", DataType::Double),
        ("decimal", dec(10, 0)),
        ("string", DataType::String),
        ("binary", DataType::Binary),
        ("date", DataType::Date),
        ("timestamp", DataType::Timestamp),
    ];
    assert_eq!(targets.len(), 12);
    for (name, target) in targets {
        let d = def(name);
        assert_eq!(d.kind, FunctionKind::Scalar, "{name}");
        assert_eq!(d.execution, Execution::Cast, "{name}");
        assert_eq!(d.inputs, Inputs::Any, "{name}");
        assert!(
            d.arity.accepts(1) && !d.arity.accepts(0) && !d.arity.accepts(2),
            "{name}"
        );
        assert_eq!(d.arity.expected(), "1", "{name}");
        // The catalog agrees (the registry takes arity from it).
        let row = builtins().spark_function(name).unwrap();
        assert_eq!(row.kind, Some(FunctionKind::Scalar), "{name}");
        // The result is the target type; NULL only with a nullable argument.
        assert_eq!(
            result(name, &[ty(target.clone())], &[]).unwrap(),
            ty(target.clone()),
            "{name}"
        );
        assert_eq!(
            result(name, &[ArgType::new(target.clone(), true)], &[]).unwrap(),
            ArgType::new(target.clone(), true),
            "{name}"
        );
        assert_eq!(
            result(name, &[null_ty()], &[]).unwrap(),
            ArgType::new(target.clone(), true),
            "{name}"
        );
    }
    // `timestamp_ntz` is not a function (Spark has no such name).
    assert!(matches!(
        builtins().lookup("timestamp_ntz", SparkProfile::V4_2),
        Lookup::NotFound
    ));
    // Lookups stay ASCII case-insensitive.
    assert!(matches!(
        builtins().lookup("FLOAT", SparkProfile::V4_2),
        Lookup::Found(_)
    ));
}

/// The percentile family is typed by the shape of its percentage: a
/// `double` (or the child's type for the approximate ones), or an array of it
/// for an array percentage; `approx_count_distinct` is a `bigint` that is
/// never NULL.
#[test]
fn quantile_results_follow_the_percentage_shape() {
    let doubles = DataType::Array(ArrayType::new(DataType::Double, false).into());
    let ints = DataType::Array(ArrayType::new(DataType::Integer, false).into());
    let pct = ty(DataType::Double);
    let pcts = ty(doubles.clone());
    let value = ty(DataType::Integer);
    {
        let name = "percentile";
        let one = result(name, &[value.clone(), pct.clone()], &[]).unwrap();
        assert_eq!(
            (one.data_type, one.nullable),
            (Some(DataType::Double), true)
        );
        let many = result(name, &[value.clone(), pcts.clone()], &[]).unwrap();
        assert_eq!(many.data_type, Some(doubles.clone()));
    }
    for name in ["percentile_approx", "approx_percentile"] {
        let one = result(name, &[value.clone(), pct.clone()], &[]).unwrap();
        assert_eq!(
            (one.data_type, one.nullable),
            (Some(DataType::Integer), true)
        );
        let many = result(name, &[value.clone(), pcts.clone()], &[]).unwrap();
        assert_eq!(many.data_type, Some(ints.clone()));
        // An untyped NULL child stays untyped NULL.
        let none = result(name, &[null_ty(), pct.clone()], &[]).unwrap();
        assert_eq!(none.data_type, Some(DataType::Null));
    }
    let median = result("median", std::slice::from_ref(&value), &[]).unwrap();
    assert_eq!(
        (median.data_type, median.nullable),
        (Some(DataType::Double), true)
    );
    for name in ["percentile_cont", "percentile_disc"] {
        let one = result(name, std::slice::from_ref(&pct), &[]).unwrap();
        assert_eq!(one.data_type, Some(DataType::Double));
        let many = result(name, std::slice::from_ref(&pcts), &[]).unwrap();
        assert_eq!(many.data_type, Some(doubles.clone()));
    }
    let distinct = result("approx_count_distinct", std::slice::from_ref(&value), &[]).unwrap();
    assert_eq!(
        (distinct.data_type, distinct.nullable),
        (Some(DataType::Long), false)
    );
}

#[test]
fn slice_wants_an_array_and_sequence_two_or_three_arguments() {
    let i = ty(DataType::Integer);
    let err = result("slice", &[i.clone(), i.clone(), i.clone()], &[]).unwrap_err();
    assert_eq!(
        err.error_class().name(),
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
    for n in [1, 4] {
        let args = vec![i.clone(); n];
        let err = result("sequence", &args, &[]).unwrap_err();
        assert_eq!(
            err.error_class().name(),
            "WRONG_NUM_ARGS.WITHOUT_SUGGESTION",
            "{n} arguments"
        );
    }
}
