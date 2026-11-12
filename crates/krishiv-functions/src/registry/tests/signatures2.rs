// SPDX-License-Identifier: Apache-2.0
//! Signature tests, part of the registry tests.

use super::*;

#[test]
fn conditional_signatures() {
    let (i, l, s) = (
        ty(DataType::Integer),
        ty(DataType::Long),
        ty(DataType::String),
    );
    // `if` and `nvl2` are NULL when a branch may be.
    let branches = [
        ty(DataType::Boolean),
        i.clone(),
        nullable(DataType::Integer),
    ];
    assert_eq!(
        result("if", &branches, &[]).unwrap(),
        nullable(DataType::Integer)
    );
    assert_eq!(result("nvl2", &[s, i.clone(), l.clone()], &[]).unwrap(), i);
    assert_eq!(
        def("if").inputs,
        Inputs::Leading(vec![InputType::Strict(DataType::Boolean)])
    );
    // `ifnull`/`nvl` are `coalesce`.
    assert_eq!(
        result("nvl", &[nullable(DataType::Long), l.clone()], &[]).unwrap(),
        l
    );
    assert_eq!(def("ifnull").inputs, Inputs::Common);
    // `nullif` keeps the first argument's type and may be NULL; `zeroifnull`
    // never is.
    assert_eq!(
        result("nullif", &[i.clone(), l], &[]).unwrap(),
        nullable(DataType::Integer)
    );
    assert_eq!(def("nullifzero").inputs, Inputs::Comparable { zero: true });
    assert_eq!(
        result("zeroifnull", std::slice::from_ref(&i), &[]).unwrap(),
        i
    );
    // `nanvl` is `float` for two floats, else `double`.
    let f = ty(DataType::Float);
    let built = |args: &[ArgType]| def("nanvl").inputs.expected(&CallTypes::new(args, &[]));
    assert_eq!(
        built(&[f.clone(), f]),
        Some(vec![InputType::Exact(DataType::Float)])
    );
    assert_eq!(
        built(&[i.clone(), i]),
        Some(vec![InputType::Exact(DataType::Double)])
    );
}

#[test]
fn first_and_last_take_a_boolean_literal_flag() {
    for (name, shown) in [
        ("first", "first"),
        ("first_value", "first"),
        ("last", "last"),
        ("last_value", "last"),
    ] {
        let d = def(name);
        assert_eq!(d.kind, FunctionKind::Aggregate);
        assert!(d.arity.accepts(1) && d.arity.accepts(2) && !d.arity.accepts(0));
        assert!(!d.arity.accepts(3));
        let agg = if shown == "first" {
            Aggregate::First
        } else {
            Aggregate::Last
        };
        assert_eq!(d.execution, Execution::Aggregate(agg));
        // The type of the value, nullable (no rows), whatever the flag.
        let flag = ty(DataType::Boolean);
        let x = ty(DataType::Integer);
        let nullable = ArgType::new(DataType::Integer, true);
        assert_eq!(
            result(name, std::slice::from_ref(&x), &[]).unwrap(),
            nullable
        );
        let consts = [None, Some(Constant::Boolean(true))];
        assert_eq!(
            result(name, &[x.clone(), flag.clone()], &consts).unwrap(),
            nullable
        );
        // Spark 4.2: anything but a boolean literal is refused.
        for consts in [
            &[None, None][..],
            &[None, Some(Constant::Null)],
            &[None, Some(Constant::Integral(1))],
            &[None, Some(Constant::String("true".to_owned()))],
        ] {
            let err = result(name, &[x.clone(), flag.clone()], consts).unwrap_err();
            assert_eq!(err.error_class(), ErrorClass::LegacyErrorTemp1210);
            assert_eq!(
                err.to_string(),
                format!("The second argument in {shown} should be a boolean literal.")
            );
        }
    }
    // An untyped NULL stays untyped.
    let r = result("first", &[null_ty()], &[]).unwrap();
    assert_eq!(r.data_type, None);
}

/// `count_if` and the boolean aggregates run as `count`, `min` and `max`
/// (Spark rewrites them so); `max_by(x, y, k)` returns an array, which
/// Krishiv has no type for, so only the two-argument form types.
#[test]
fn simple_aggregates() {
    for (name, agg) in [
        ("count_if", Aggregate::Count),
        ("bool_and", Aggregate::Min),
        ("every", Aggregate::Min),
        ("bool_or", Aggregate::Max),
        ("some", Aggregate::Max),
        ("any", Aggregate::Max),
        ("any_value", Aggregate::AnyValue),
        ("max_by", Aggregate::MaxBy),
        ("min_by", Aggregate::MinBy),
    ] {
        let d = def(name);
        assert_eq!(d.kind, FunctionKind::Aggregate, "{name}");
        assert_eq!(d.execution, Execution::Aggregate(agg), "{name}");
        assert!(d.deterministic, "{name}");
    }
    let boolean = Some(vec![InputType::Exact(DataType::Boolean)]);
    let expects = |name: &str| def(name).inputs.expected(&CallTypes::new(&[], &[]));
    let bool_result = ArgType::new(DataType::Boolean, true);
    for name in ["count_if", "bool_and", "every", "bool_or", "some", "any"] {
        assert_eq!(expects(name), boolean, "{name}");
        assert!(def(name).arity.accepts(1) && !def(name).arity.accepts(2));
    }
    let x = ty(DataType::Boolean);
    // `count_if` is `bigint` and never NULL; the others are nullable
    // `boolean`s whatever their argument.
    assert_eq!(
        result("count_if", std::slice::from_ref(&x), &[]).unwrap(),
        ArgType::new(DataType::Long, false)
    );
    for name in ["bool_and", "every", "bool_or", "some", "any"] {
        assert_eq!(
            result(name, std::slice::from_ref(&x), &[]).unwrap(),
            bool_result
        );
    }
    // `max_by(x, y)` has the type of `x`, nullable.
    let (v, k) = (ty(DataType::String), ty(DataType::Integer));
    for name in ["max_by", "min_by"] {
        let d = def(name);
        assert!(d.arity.accepts(2) && d.arity.accepts(3) && !d.arity.accepts(1));
        assert_eq!(
            result(name, &[v.clone(), k.clone()], &[]).unwrap(),
            ArgType::new(DataType::String, true)
        );
        let err = result(name, &[v.clone(), k.clone(), ty(DataType::Integer)], &[]).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    }
    // `any_value(x [, ignoreNulls])` takes the flag as `first` does.
    let nullable = ArgType::new(DataType::Integer, true);
    assert_eq!(
        result("any_value", &[ty(DataType::Integer)], &[]).unwrap(),
        nullable
    );
    let err = result(
        "any_value",
        &[ty(DataType::Integer), ty(DataType::Boolean)],
        &[None, None],
    )
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "The second argument in any_value should be a boolean literal."
    );
    // `collect_list`, `collect_set` and `array_agg` are aggregates over
    // one argument of any type; the result is an array of the argument's
    // type, never NULL. `vector_sum`/`vector_avg` take `ARRAY<FLOAT>` and
    // give an array of `float`, NULL when no non-NULL vector was seen.
    for (name, agg) in [
        ("collect_list", Aggregate::CollectList),
        ("collect_set", Aggregate::CollectSet),
        ("array_agg", Aggregate::CollectList),
    ] {
        let d = def(name);
        assert_eq!(d.execution, Execution::Aggregate(agg), "{name}");
        assert_eq!(d.kind, FunctionKind::Aggregate, "{name}");
        assert!(d.deterministic, "{name}");
        assert!(d.arity.accepts(1) && !d.arity.accepts(2), "{name}");
        let array = DataType::Array(ArrayType::new(DataType::Integer, false).into());
        assert_eq!(
            result(name, &[ty(DataType::Integer)], &[]).unwrap(),
            ArgType::new(array, false),
            "{name}"
        );
    }
    // Spark refuses a map anywhere inside the input of `collect_set` only.
    let map = DataType::Map(
        krishiv_types::MapType::new(DataType::Integer, DataType::Integer, true).into(),
    );
    for nested in [
        map.clone(),
        DataType::Array(ArrayType::new(map, true).into()),
    ] {
        let err = result("collect_set", &[ty(nested.clone())], &[]).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchUnsupportedInputType,
            "{nested:?}"
        );
        assert!(result("collect_list", &[ty(nested.clone())], &[]).is_ok());
        assert!(result("array_agg", &[ty(nested)], &[]).is_ok());
    }
    for (name, agg) in [
        ("vector_sum", Aggregate::VectorSum),
        ("vector_avg", Aggregate::VectorAvg),
    ] {
        let d = def(name);
        assert_eq!(d.execution, Execution::Aggregate(agg), "{name}");
        assert_eq!(d.kind, FunctionKind::Aggregate, "{name}");
        assert!(d.arity.accepts(1) && !d.arity.accepts(2), "{name}");
        let floats = || DataType::Array(ArrayType::new(DataType::Float, false).into());
        assert_eq!(
            result(name, &[ty(floats())], &[]).unwrap(),
            ArgType::new(floats(), true),
            "{name}"
        );
        let err = result(name, &[ty(DataType::Integer)], &[]).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{name}"
        );
    }
}

#[test]
fn session_and_clock_functions() {
    for name in ["current_user", "user", "session_user"] {
        let d = def(name);
        assert_eq!(d.execution, Execution::Scalar);
        assert!(d.arity.accepts(0) && !d.arity.accepts(1));
        assert_eq!(result(name, &[], &[]).unwrap(), ty(DataType::String));
    }
    // `session_user` arrived in 4.0.
    assert!(matches!(
        builtins().lookup("session_user", SparkProfile::V3_5),
        Lookup::NotFound
    ));
    assert_eq!(
        result("localtimestamp", &[], &[]).unwrap(),
        ty(DataType::TimestampNtz)
    );
    // `current_time` is a `TIME`, evaluated since P2-T181.
    assert!(matches!(
        builtins().lookup("current_time", SparkProfile::V4_2),
        Lookup::Found(_)
    ));
}

/// The misc and seeded-random signatures: arities, inputs and result types.
#[test]
fn misc_signatures() {
    let (s, i, l) = (
        ty(DataType::String),
        ty(DataType::Integer),
        ty(DataType::Long),
    );
    // The session constants and `version` take no arguments and are never
    // NULL.
    for name in [
        "version",
        "current_database",
        "current_schema",
        "current_catalog",
        "input_file_name",
        "monotonically_increasing_id",
        "spark_partition_id",
        "typeof",
        "uuid",
        "rand",
        "random",
        "randn",
    ] {
        let d = def(name);
        assert_eq!(d.kind, FunctionKind::Scalar, "{name}");
        assert_eq!(d.execution, Execution::Scalar, "{name}");
    }
    for name in [
        "version",
        "current_database",
        "current_schema",
        "current_catalog",
    ] {
        assert_eq!(result(name, &[], &[]).unwrap(), s, "{name}");
        assert!(def(name).deterministic, "{name}");
    }
    // The file placeholders and partition functions are nondeterministic.
    for name in [
        "input_file_name",
        "input_file_block_start",
        "input_file_block_length",
        "monotonically_increasing_id",
        "spark_partition_id",
    ] {
        assert!(!def(name).deterministic, "{name}");
    }
    assert_eq!(result("input_file_name", &[], &[]).unwrap(), s);
    assert_eq!(
        result("input_file_block_start", &[], &[]).unwrap(),
        ty(DataType::Long)
    );
    assert_eq!(
        result("input_file_block_length", &[], &[]).unwrap(),
        ty(DataType::Long)
    );
    assert_eq!(result("monotonically_increasing_id", &[], &[]).unwrap(), l);
    assert_eq!(result("spark_partition_id", &[], &[]).unwrap(), i);
    assert_eq!(result("typeof", std::slice::from_ref(&s), &[]).unwrap(), s);
    // Never NULL, even for a nullable or untyped-NULL argument.
    assert_eq!(result("typeof", &[null_ty()], &[]).unwrap(), s);
    assert_eq!(
        result("typeof", &[ArgType::new(DataType::String, true)], &[]).unwrap(),
        s
    );
    assert_eq!(result("uuid", &[], &[]).unwrap(), s);
    assert_eq!(result("rand", &[], &[]).unwrap(), ty(DataType::Double));
    assert_eq!(
        result(
            "rand",
            std::slice::from_ref(&l),
            &[Some(Constant::Integral(0))]
        )
        .unwrap(),
        ty(DataType::Double)
    );
    // A NULL seed reads as 0.
    assert_eq!(
        result("rand", &[null_ty()], &[Some(Constant::Null)]).unwrap(),
        ty(DataType::Double)
    );
    // A short seed is the wrong type; a non-foldable one is refused.
    let err = result(
        "rand",
        &[ty(DataType::Short)],
        &[Some(Constant::Integral(1))],
    )
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(err.to_string().contains("(\"INT\" or \"BIGINT\")"), "{err}");
    let err = result("rand", std::slice::from_ref(&i), &[]).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::SeedExpressionIsUnfoldable);
    assert!(err.to_string().contains("must be foldable"), "{err}");
    // `uuid` wants a plain literal.
    let err = result("uuid", std::slice::from_ref(&i), &[]).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidParameterValueLong);
    assert!(err.to_string().contains("`UUID`"), "{err}");
    // A non-integral literal is the wrong seed too.
    let err = result(
        "uuid",
        &[ty(DataType::Double)],
        &[Some(Constant::Fractional(1))],
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidParameterValueLong);
}

/// `uniform` and `randstr`: the golden types and errors.
#[test]
fn uniform_and_randstr_signatures() {
    let (i, l) = (ty(DataType::Integer), ty(DataType::Long));
    let (ten, twenty, zero) = (
        Some(Constant::Integral(10)),
        Some(Constant::Integral(20)),
        Some(Constant::Integral(0)),
    );
    // Integral bounds widen; the seed does not change the type.
    assert_eq!(
        result(
            "uniform",
            &[i.clone(), i.clone(), i.clone()],
            &[ten.clone(), twenty.clone(), zero.clone()]
        )
        .unwrap(),
        i
    );
    assert_eq!(
        result(
            "uniform",
            &[i.clone(), l.clone(), l.clone()],
            &[ten.clone(), twenty.clone(), zero.clone()]
        )
        .unwrap(),
        l
    );
    // A NULL bound is a NULL double; a NULL seed reads as 0.
    assert_eq!(
        result(
            "uniform",
            &[null_ty(), i.clone(), i.clone()],
            &[Some(Constant::Null), twenty.clone(), zero.clone()]
        )
        .unwrap(),
        ArgType::new(DataType::Double, true)
    );
    // Narrower integrals widen; decimals meet at the wider.
    assert_eq!(
        result(
            "uniform",
            &[ty(DataType::Byte), ty(DataType::Short)],
            &[Some(Constant::Integral(1)), Some(Constant::Integral(2))]
        )
        .unwrap(),
        ty(DataType::Short)
    );
    assert_eq!(
        result(
            "uniform",
            &[ty(dec(10, 3)), ty(dec(11, 4))],
            &[Some(Constant::Fractional(1)), Some(Constant::Fractional(2))]
        )
        .unwrap(),
        ty(dec(11, 4))
    );
    assert_eq!(
        result(
            "uniform",
            &[ty(DataType::Float), ty(DataType::Double)],
            &[Some(Constant::Fractional(1)), Some(Constant::Fractional(2))]
        )
        .unwrap(),
        ty(DataType::Double)
    );
    assert_eq!(
        result(
            "uniform",
            &[i.clone(), i.clone(), null_ty()],
            &[ten.clone(), twenty.clone(), Some(Constant::Null)]
        )
        .unwrap(),
        ArgType::new(DataType::Integer, true)
    );
    // A string bound is the wrong type; a column seed is not foldable.
    let s = ty(DataType::String);
    let err = result(
        "uniform",
        &[s, i.clone(), i.clone()],
        &[None, twenty.clone(), zero.clone()],
    )
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(err.to_string().contains("\"NUMERIC\""), "{err}");
    let err = result(
        "uniform",
        &[i.clone(), i.clone(), i.clone()],
        &[ten.clone(), twenty.clone(), None],
    )
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    assert!(err.to_string().contains("`seed`"), "{err}");
    // A failing fold propagates instead of typing the call.
    let failed = Constant::Error(KrishivError::from(krishiv_common::error::InternalError {
        message: "boom".to_owned(),
    }));
    let err = result(
        "uniform",
        &[i.clone(), i.clone(), i.clone()],
        &[ten.clone(), twenty, Some(failed.clone())],
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    let err = result(
        "randstr",
        &[i.clone(), i.clone()],
        &[ten.clone(), Some(failed)],
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    // `randstr` is a string; a boolean length is the wrong type.
    assert_eq!(
        result("randstr", &[i.clone(), i.clone()], &[ten, zero]).unwrap(),
        ty(DataType::String)
    );
    let err = result(
        "randstr",
        &[ty(DataType::Boolean), i],
        &[Some(Constant::Boolean(true)), None],
    )
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(err.to_string().contains("\"INT\""), "{err}");
}

#[test]
fn uniform_result_agrees_with_the_executor_bind() {
    use crate::scalar::{BindOptions, bind_scalar};
    // One foldable constant per bound type, as the analyzer sees them.
    let cases: Vec<(Option<DataType>, Option<Constant>)> = vec![
        (Some(DataType::Byte), Some(Constant::Integral(1))),
        (Some(DataType::Short), Some(Constant::Integral(1))),
        (Some(DataType::Integer), Some(Constant::Integral(10))),
        (Some(DataType::Long), Some(Constant::Integral(20))),
        (Some(DataType::Float), Some(Constant::Fractional(1))),
        (Some(DataType::Double), Some(Constant::Fractional(1))),
        (Some(dec(10, 3)), Some(Constant::Fractional(1))),
        (Some(dec(11, 4)), Some(Constant::Fractional(1))),
        (None, Some(Constant::Null)),
    ];
    for (a, ca) in &cases {
        for (b, cb) in &cases {
            let args = [
                ArgType {
                    data_type: a.clone(),
                    nullable: a.is_none(),
                },
                ArgType {
                    data_type: b.clone(),
                    nullable: b.is_none(),
                },
            ];
            let reg = result("uniform", &args, &[ca.clone(), cb.clone()]).unwrap();
            let bound = bind_scalar(
                "uniform",
                &[
                    wide(&a.clone().unwrap_or(DataType::Null)).into(),
                    wide(&b.clone().unwrap_or(DataType::Null)).into(),
                ],
                &BindOptions::new(),
            )
            .unwrap();
            assert_eq!(
                reg.data_type.as_ref().map(wide),
                Some(bound.return_type().clone()),
                "uniform({a:?}, {b:?})"
            );
        }
    }
}

#[test]
fn bitwise_signatures() {
    let (b, s, i, l) = (
        ty(DataType::Byte),
        ty(DataType::Short),
        ty(DataType::Integer),
        ty(DataType::Long),
    );
    let built =
        |name: &str, args: &[ArgType]| def(name).inputs.expected(&CallTypes::new(args, &[]));
    // `&` widens integrals, meets strings as `bigint`, refuses the rest.
    assert_eq!(
        built("&", &[b.clone(), s]),
        Some(vec![
            InputType::Exact(DataType::Short),
            InputType::Exact(DataType::Short)
        ])
    );
    assert_eq!(
        built("|", &[i.clone(), l.clone()]),
        Some(vec![
            InputType::Exact(DataType::Long),
            InputType::Exact(DataType::Long)
        ])
    );
    assert_eq!(
        result("^", &[i.clone(), i.clone()], &[]).unwrap(),
        ty(DataType::Integer)
    );
    // An untyped `NULL` meets anything as itself (`int` for two nulls).
    assert_eq!(
        built("&", &[null_ty(), null_ty()]),
        Some(vec![
            InputType::Exact(DataType::Integer),
            InputType::Exact(DataType::Integer)
        ])
    );
    assert_eq!(
        built("|", &[null_ty(), l.clone()]),
        Some(vec![
            InputType::Exact(DataType::Long),
            InputType::Exact(DataType::Long)
        ])
    );
    assert_eq!(
        result("&", &[i.clone(), l.clone()], &[]).unwrap(),
        ty(DataType::Long)
    );
    assert_eq!(
        result("&", &[nullable(DataType::Long), l.clone()], &[]).unwrap(),
        nullable(DataType::Long)
    );
    // A string meets integrals as `bigint` and non-integral numbers as
    // `double` (never narrowed through `bigint`), which is then the wrong
    // type: Spark 4.2 raises BINARY_OP_WRONG_TYPE on DOUBLE for
    // `'5' & 1.5D`, `'5' & 1.5BD` and `'5' & CAST(1 AS FLOAT)`.
    assert_eq!(
        built("&", &[ty(DataType::String), l.clone()]),
        Some(vec![
            InputType::Exact(DataType::Long),
            InputType::Exact(DataType::Long)
        ])
    );
    for t in [DataType::Float, DataType::Double, dec(10, 2)] {
        assert_eq!(
            built("&", &[ty(DataType::String), ty(t.clone())]),
            Some(vec![
                InputType::Exact(DataType::Double),
                InputType::Exact(DataType::Double)
            ])
        );
    }
    // Two operands of one non-integral type are the wrong type, not a
    // mismatch (Spark 4.2: `1.5D & 2.5D`).
    let err = result("&", &[ty(DataType::Double), ty(DataType::Double)], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchBinaryOpWrongType
    );
    assert!(err.to_string().contains("\"DOUBLE\""), "{err}");
    // `~` keeps its child's type; strings are refused as `INTEGRAL`.
    assert_eq!(result("~", std::slice::from_ref(&b), &[]).unwrap(), b);
    assert_eq!(result("~", std::slice::from_ref(&l), &[]).unwrap(), l);
    assert_eq!(
        result("~", &[null_ty()], &[]).unwrap(),
        ArgType::new(DataType::Integer, true)
    );
    let err = result("~", &[ty(DataType::String)], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(err.to_string().contains("\"INTEGRAL\""), "{err}");
    // Shifts keep `int` (or `bigint` for a `bigint`), by an `int`.
    assert_eq!(
        result("shiftleft", &[i.clone(), i.clone()], &[]).unwrap(),
        i
    );
    assert_eq!(
        result(">>", &[l.clone(), i.clone()], &[]).unwrap(),
        ty(DataType::Long)
    );
    // Nullability passes through the result: a nullable operand stays
    // nullable, as the analyzer records it.
    assert_eq!(
        result(">>>", &[nullable(DataType::Integer), i.clone()], &[]).unwrap(),
        nullable(DataType::Integer)
    );
    // A bad left side is refused with `("INT" or "BIGINT")` (checked by
    // the inputs; the matrix covers the error class end to end).
    assert_eq!(
        built("<<", &[ty(DataType::Boolean), i.clone()]),
        Some(vec![
            InputType::OneOf(vec![DataType::Integer, DataType::Long]),
            InputType::Exact(DataType::Integer)
        ])
    );
    // `bit_count` takes integrals and booleans, always an `int`.
    assert_eq!(
        result("bit_count", std::slice::from_ref(&b), &[]).unwrap(),
        i
    );
    assert_eq!(
        result("bit_count", &[ty(DataType::Boolean)], &[]).unwrap(),
        ty(DataType::Integer)
    );
    assert_eq!(
        result("bit_count", &[null_ty()], &[]).unwrap(),
        nullable(DataType::Integer)
    );
    let err = result("bit_count", &[ty(DataType::Double)], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(
        err.to_string().contains("(\"INTEGRAL\" or \"BOOLEAN\")"),
        "{err}"
    );
    // `bit_get`/`getbit` keep the value's width for the range check and
    // return a `tinyint`.
    assert_eq!(
        result("bit_get", &[l, i.clone()], &[]).unwrap(),
        ty(DataType::Byte)
    );
    assert_eq!(
        result("getbit", &[i.clone(), i.clone()], &[]).unwrap(),
        ty(DataType::Byte)
    );
    assert_eq!(
        result("getbit", &[i.clone(), nullable(DataType::Integer)], &[]).unwrap(),
        nullable(DataType::Byte)
    );
    let err = result("getbit", &[ty(DataType::String), i], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(err.to_string().contains("\"INTEGRAL\""), "{err}");
}

#[test]
fn predicate_signatures() {
    let (i, l, s) = (
        ty(DataType::Integer),
        ty(DataType::Long),
        ty(DataType::String),
    );
    // `equal_null` is a boolean, never NULL, whatever the sides.
    assert_eq!(
        result("equal_null", &[i.clone(), l], &[]).unwrap(),
        ty(DataType::Boolean)
    );
    assert_eq!(
        result("equal_null", &[null_ty(), null_ty()], &[]).unwrap(),
        ty(DataType::Boolean)
    );
    // `isnan` takes floats and doubles (anything else Spark casts is a
    // double), never NULL.
    assert_eq!(
        result("isnan", &[ty(DataType::Float)], &[]).unwrap(),
        ty(DataType::Boolean)
    );
    assert_eq!(
        result("isnan", &[nullable(DataType::Double)], &[]).unwrap(),
        ty(DataType::Boolean)
    );
    let built =
        |name: &str, args: &[ArgType]| def(name).inputs.expected(&CallTypes::new(args, &[]));
    assert_eq!(
        built("isnan", &[ty(DataType::Float)]),
        Some(vec![InputType::Exact(DataType::Float)])
    );
    assert_eq!(
        built("isnan", std::slice::from_ref(&i)),
        Some(vec![InputType::Exact(DataType::Double)])
    );
    // `isnull`/`isnotnull` are booleans, never NULL.
    assert_eq!(
        result("isnull", &[null_ty()], &[]).unwrap(),
        ty(DataType::Boolean)
    );
    assert_eq!(
        result("isnotnull", std::slice::from_ref(&s), &[]).unwrap(),
        ty(DataType::Boolean)
    );
    // `assert_true` is `void`, always nullable; the condition is strictly
    // boolean.
    assert_eq!(
        result("assert_true", &[ty(DataType::Boolean), s.clone()], &[]).unwrap(),
        ArgType {
            data_type: None,
            nullable: true,
        }
    );
    assert_eq!(
        def("assert_true").inputs,
        Inputs::Expects(vec![
            InputType::Strict(DataType::Boolean),
            InputType::Exact(DataType::String)
        ])
    );
    // `raise_error` is `void`, always nullable; the message is a string.
    assert_eq!(
        result("raise_error", std::slice::from_ref(&s), &[]).unwrap(),
        ArgType {
            data_type: None,
            nullable: true,
        }
    );
    assert_eq!(def("raise_error").execution, Execution::Scalar);
}

#[test]
fn number_signatures() {
    let (s, i, l) = (
        ty(DataType::String),
        ty(DataType::Integer),
        ty(DataType::Long),
    );
    // `hex` keeps binaries and strings, takes longs for numbers.
    let built =
        |name: &str, args: &[ArgType]| def(name).inputs.expected(&CallTypes::new(args, &[]));
    assert_eq!(
        built("hex", std::slice::from_ref(&i)),
        Some(vec![InputType::Exact(DataType::Long)])
    );
    assert_eq!(
        built("hex", &[ty(DataType::Binary)]),
        Some(vec![InputType::Exact(DataType::Binary)])
    );
    assert_eq!(
        built("hex", &[ty(DataType::Boolean)]),
        Some(vec![InputType::Exact(DataType::String)])
    );
    assert_eq!(
        result("hex", std::slice::from_ref(&l), &[]).unwrap(),
        ty(DataType::String)
    );
    assert_eq!(
        result("hex", &[nullable(DataType::Double)], &[]).unwrap(),
        nullable(DataType::String)
    );
    // `unhex` is binary, always nullable; `bin` is a string.
    assert_eq!(
        result("unhex", std::slice::from_ref(&s), &[]).unwrap(),
        ArgType::new(DataType::Binary, true)
    );
    assert_eq!(
        result("bin", std::slice::from_ref(&l), &[]).unwrap(),
        ty(DataType::String)
    );
    assert_eq!(
        result("bin", &[nullable(DataType::Double)], &[]).unwrap(),
        nullable(DataType::String)
    );
    // `conv` is a string, always nullable.
    assert_eq!(
        result("conv", &[s.clone(), i.clone(), i.clone()], &[]).unwrap(),
        ArgType::new(DataType::String, true)
    );
    // `format_number` is a string, always nullable; the second argument
    // is an `int` or a string, the first numeric.
    assert_eq!(
        result("format_number", &[ty(DataType::Double), i.clone()], &[]).unwrap(),
        ArgType::new(DataType::String, true)
    );
    let err = result("format_number", &[s.clone(), i], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(err.to_string().contains("\"NUMERIC\""), "{err}");
    let err = result("format_number", &[ty(DataType::Double), l], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(err.to_string().contains("(\"INT\" or \"STRING\")"), "{err}");
    // `to_number` types the pattern; `try_to_number` is always nullable.
    let fmt = |p: &str| [None, Some(Constant::String(p.to_owned()))];
    let args = [s.clone(), s.clone()];
    assert_eq!(
        result("to_number", &args, &fmt("000")).unwrap(),
        ArgType::new(dec(3, 0), false)
    );
    assert_eq!(
        result("to_number", &args, &fmt("$00.00")).unwrap(),
        ArgType::new(dec(4, 2), false)
    );
    assert_eq!(
        result("try_to_number", &args, &fmt("000")).unwrap(),
        ArgType::new(dec(3, 0), true)
    );
    assert_eq!(
        result(
            "to_number",
            &[s.clone(), nullable(DataType::String)],
            &[None, Some(Constant::Null)]
        )
        .unwrap(),
        ArgType::new(dec(10, 0), true)
    );
    for bad in ["", "1", "00$00", "00..00"] {
        let err = result("to_number", &args, &fmt(bad)).unwrap_err();
        assert!(
            matches!(
                err.error_class(),
                ErrorClass::InvalidFormatEmpty
                    | ErrorClass::InvalidFormatWrongNumDigit
                    | ErrorClass::InvalidFormatCurMustBeforeDigit
                    | ErrorClass::InvalidFormatWrongNumToken
            ),
            "{bad}: {err}"
        );
    }
    let err = result("to_number", &args, &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    // `to_char`/`to_varchar` are strings; bad values are `"DECIMAL"`.
    assert_eq!(
        result("to_char", &args, &fmt("000")).unwrap(),
        ty(DataType::String)
    );
    let err = result("to_char", &[ty(DataType::Boolean), s.clone()], &fmt("000")).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(err.to_string().contains("\"DECIMAL\""), "{err}");
    // A binary `to_char` only takes `base64`/`hex`/`utf-8`.
    let err = result("to_char", &[ty(DataType::Binary), s], &fmt("999")).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::InvalidParameterValueBinaryFormat
    );
}

#[test]
fn number_formats_render_through_casts() {
    let s = ty(DataType::String);
    let args = [ty(DataType::String), s];
    let at = |sql: &str| {
        let with_sql = |n: usize| (n == 1).then(|| sql.to_owned());
        (def("to_number").result)(&CallTypes::new(&args, &[None, None]).with_sql(&with_sql))
    };
    // `CAST('999' AS STRING)` renders through the cast.
    assert_eq!(
        at("CAST('999' AS STRING)").unwrap(),
        ArgType::new(dec(3, 0), false)
    );
    // Casts of numbers render as Spark casts them to strings (`1.50`,
    // `1`, `1.5`): none is a valid pattern, and each fails its own way.
    let err = at("CAST(1.5 AS DECIMAL(10,2))").unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidFormatUnexpectedToken);
    for sql in [
        "CAST(1 AS SMALLINT)",
        "CAST(1 AS TINYINT)",
        "CAST(1.5 AS FLOAT)",
    ] {
        let err = at(sql).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::InvalidFormatWrongNumDigit,
            "{sql}"
        );
    }
    // Escapes render through the cast.
    let with_sql = |n: usize| (n == 1).then(|| "CAST('a\\\'b\nc' AS STRING)".to_owned());
    assert_eq!(
        (def("to_number").result)(&CallTypes::new(&args, &[None, None]).with_sql(&with_sql))
            .unwrap_err()
            .error_class(),
        ErrorClass::InvalidFormatWrongNumDigit
    );
    // A non-foldable, non-literal format is refused.
    let with_sql = |n: usize| (n == 1).then(|| "concat('9', '9')".to_owned());
    let err = (def("to_number").result)(&CallTypes::new(&args, &[None, None]).with_sql(&with_sql))
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    // A failing cast propagates its error.
    let failed = Constant::Error(KrishivError::from(
        krishiv_common::error::CastInvalidInput {
            expression: "'abc'".to_owned(),
            source_type: "\"STRING\"".to_owned(),
            target_type: "\"INT\"".to_owned(),
        },
    ));
    let with_sql = |n: usize| (n == 1).then(|| "CAST('abc' AS INT)".to_owned());
    let err = (def("to_number").result)(
        &CallTypes::new(&args, &[None, Some(failed)]).with_sql(&with_sql),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::CastInvalidInput);
    // `hex` of a timestamp renders through the string cast.
    assert_eq!(
        result("hex", &[ty(DataType::Timestamp)], &[]).unwrap(),
        ty(DataType::String)
    );
}

/// The regular-expression functions: arities, inputs and result types.
#[test]
fn regex_signatures() {
    let s = ty(DataType::String);
    let exact = InputType::Exact;
    let strings = || Inputs::Expects(vec![exact(DataType::String), exact(DataType::String)]);
    // `regexp` and `regexp_like` take two strings and may be NULL.
    for name in ["regexp", "regexp_like"] {
        let d = def(name);
        assert!(d.arity.accepts(2) && !d.arity.accepts(1) && !d.arity.accepts(3));
        assert_eq!(d.inputs, strings());
        assert_eq!(
            result(name, &[s.clone(), s.clone()], &[]).unwrap(),
            ty(DataType::Boolean)
        );
        assert_eq!(
            result(
                name,
                &[ArgType::new(DataType::String, true), s.clone()],
                &[]
            )
            .unwrap(),
            ArgType::new(DataType::Boolean, true)
        );
    }
    // The two- and three-argument input shapes.
    let indexed = || {
        Inputs::Expects(vec![
            exact(DataType::String),
            exact(DataType::String),
            exact(DataType::Integer),
        ])
    };
    assert_eq!(def("regexp_substr").inputs, strings());
    assert_eq!(def("regexp_count").inputs, strings());
    assert_eq!(def("regexp_extract").inputs, indexed());
    assert_eq!(def("regexp_instr").inputs, indexed());
    for (name, ty) in [
        ("regexp_extract", DataType::String),
        ("regexp_count", DataType::Integer),
        ("regexp_instr", DataType::Integer),
    ] {
        let d = def(name);
        assert!(d.arity.accepts(2), "{name}");
        assert_eq!(
            result(name, &[s.clone(), s.clone()], &[]).unwrap(),
            ArgType::new(ty, false)
        );
    }
    // `regexp_substr` is nullable whatever its arguments.
    assert_eq!(
        result("regexp_substr", &[s.clone(), s.clone()], &[]).unwrap(),
        ArgType::new(DataType::String, true)
    );
    assert!(def("regexp_extract").arity.accepts(3));
    assert!(!def("regexp_extract").arity.accepts(4));
    assert!(!def("regexp_substr").arity.accepts(3));
    assert!(!def("regexp_count").arity.accepts(3));
    // A NULL string is a NULL result.
    let r = result("regexp_like", &[null_ty(), s.clone()], &[]).unwrap();
    assert_eq!(r, ArgType::new(DataType::Boolean, true));
    let r = result("regexp_count", &[s, null_ty()], &[]).unwrap();
    assert_eq!(r, ArgType::new(DataType::Integer, true));
}

#[test]
fn json_signatures() {
    use krishiv_types::{MapType, StructField, StructType};

    let s = ty(DataType::String);
    let map_ss = || DataType::Map(MapType::new(DataType::String, DataType::String, true).into());
    let string_lit = |text: &str| Some(Constant::String(text.to_owned()));
    // `json_array_length` takes a string and nothing else; `get_json_object`
    // takes strings exactly.
    assert_eq!(
        def("json_array_length").inputs,
        Inputs::Expects(vec![InputType::Strict(DataType::String)])
    );
    assert_eq!(
        result("json_array_length", std::slice::from_ref(&s), &[]).unwrap(),
        ArgType::new(DataType::Integer, true)
    );
    assert_eq!(
        result("get_json_object", &[s.clone(), s.clone()], &[]).unwrap(),
        ArgType::new(DataType::String, true)
    );
    // `json_object_keys` reads an array of non-null strings, always nullable.
    assert_eq!(
        result("json_object_keys", std::slice::from_ref(&s), &[]).unwrap(),
        ArgType::new(
            DataType::Array(ArrayType::new(DataType::String, false).into()),
            true,
        )
    );
    // `schema_of_json` needs a foldable string: a null literal and a column fail.
    assert_eq!(
        result("schema_of_json", &[null_ty()], &[Some(Constant::Null)])
            .unwrap_err()
            .error_class(),
        ErrorClass::DatatypeMismatchUnexpectedNull
    );
    assert_eq!(
        result("schema_of_json", std::slice::from_ref(&s), &[])
            .unwrap_err()
            .error_class(),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    assert_eq!(
        result(
            "schema_of_json",
            std::slice::from_ref(&s),
            &[string_lit("{\"a\":1}")]
        )
        .unwrap(),
        ArgType::new(DataType::String, false)
    );
    // `from_json` reads its schema: a bare field list wraps in a struct.
    let struct_ab = || {
        DataType::Struct(StructType::new(vec![
            StructField::new("a", DataType::Integer, true),
            StructField::new("b", DataType::Double, true),
        ]))
    };
    let from = |schema: Option<Constant>| {
        result(
            "from_json",
            &[s.clone(), s.clone()],
            &[string_lit("x"), schema],
        )
    };
    assert_eq!(
        from(string_lit("a INT, b DOUBLE")).unwrap(),
        ArgType::new(struct_ab(), true)
    );
    assert_eq!(
        from(string_lit("STRUCT<a: INT, b: DOUBLE>")).unwrap(),
        ArgType::new(struct_ab(), true)
    );
    assert_eq!(
        from(string_lit("map<string, int>")).unwrap(),
        ArgType::new(
            DataType::Map(MapType::new(DataType::String, DataType::Integer, true).into()),
            true,
        )
    );
    for bad in [
        Some(Constant::Null),
        Some(Constant::Integral(1)),
        Some(Constant::Boolean(true)),
        None,
    ] {
        assert_eq!(
            from(bad).unwrap_err().error_class(),
            ErrorClass::InvalidSchemaNonStringLiteral
        );
    }
    assert_eq!(
        from(string_lit("a InvalidType")).unwrap_err().error_class(),
        ErrorClass::ParseSyntaxError
    );
    // A non-string document fails with the single-argument call text.
    assert_eq!(
        result(
            "from_json",
            &[ty(DataType::Integer), s.clone()],
            &[None, string_lit("a INT")]
        )
        .unwrap_err()
        .error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    // The options must be a non-null string map.
    let with_options = |options: ArgType, constant: Option<Constant>| {
        result(
            "from_json",
            &[s.clone(), s.clone(), options],
            &[string_lit("x"), string_lit("a INT"), constant],
        )
    };
    assert_eq!(
        with_options(
            ArgType::new(map_ss(), true),
            Some(Constant::String("x".to_owned()))
        )
        .unwrap(),
        ArgType::new(
            DataType::Struct(StructType::new(vec![StructField::new(
                "a",
                DataType::Integer,
                true
            )])),
            true,
        )
    );
    assert_eq!(
        with_options(ArgType::new(map_ss(), true), Some(Constant::Null))
            .unwrap_err()
            .error_class(),
        ErrorClass::InvalidOptionsNonMapFunction
    );
    assert_eq!(
        with_options(ty(DataType::String), string_lit("x"))
            .unwrap_err()
            .error_class(),
        ErrorClass::InvalidOptionsNonMapFunction
    );
    assert_eq!(
        with_options(
            ArgType::new(
                DataType::Map(MapType::new(DataType::String, DataType::Integer, true).into()),
                true,
            ),
            string_lit("x")
        )
        .unwrap_err()
        .error_class(),
        ErrorClass::InvalidOptionsNonStringType
    );
    // `to_json` only takes structs, maps and arrays.
    assert_eq!(
        result("to_json", &[ArgType::new(struct_ab(), true)], &[]).unwrap(),
        ArgType::new(DataType::String, true)
    );
    assert_eq!(
        result("to_json", &[ty(DataType::Integer)], &[])
            .unwrap_err()
            .error_class(),
        ErrorClass::DatatypeMismatchInvalidJsonSchema
    );
    // A folded error propagates.
    let err: KrishivError = InvalidOptionsNonMapFunction {}.into();
    assert_eq!(
        result(
            "schema_of_json",
            std::slice::from_ref(&s),
            &[Some(Constant::Error(err))]
        )
        .unwrap_err()
        .error_class(),
        ErrorClass::InvalidOptionsNonMapFunction
    );
    // A folded error in the schema position propagates from `from_json`.
    let err: KrishivError = InvalidOptionsNonMapFunction {}.into();
    let failed = Some(Constant::Error(err));
    assert_eq!(
        result(
            "from_json",
            &[s.clone(), s.clone()],
            &[string_lit("x"), failed]
        )
        .unwrap_err()
        .error_class(),
        ErrorClass::InvalidOptionsNonMapFunction
    );
    // An empty or blank schema reports the direct parse error.
    for schema in ["", "   "] {
        assert_eq!(
            from(string_lit(schema)).unwrap_err().error_class(),
            ErrorClass::ParseEmptyStatement
        );
    }
}
