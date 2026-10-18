// SPDX-License-Identifier: Apache-2.0

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_types::{DataType, DecimalType};

use super::{Aggregate, Execution, FunctionDef, Lookup, builtins};
use crate::scalar::scalar_names;
use crate::signature::{
    ArgType, CallTypes, Constant, FunctionKind, InputType, Inputs, Operator, SparkProfile,
};

fn def(name: &str) -> &'static FunctionDef {
    match builtins().lookup(name, SparkProfile::default()) {
        Lookup::Found(d) => d,
        other => panic!("{name}: {other:?}"),
    }
}

fn result(
    name: &str,
    args: &[ArgType],
    consts: &[Option<Constant>],
) -> Result<ArgType, KrishivError> {
    (def(name).result)(&CallTypes::new(args, consts))
}

fn ty(t: DataType) -> ArgType {
    ArgType::new(t, false)
}

fn null_ty() -> ArgType {
    ArgType {
        data_type: None,
        nullable: true,
    }
}

fn dec(p: u32, s: u32) -> DataType {
    DataType::Decimal(DecimalType::try_new(p, s).unwrap())
}

#[test]
fn every_definition_is_in_sparks_catalog() {
    let r = builtins();
    assert!(r.unlisted().is_empty(), "{:?}", r.unlisted());
    assert_eq!(r.spark_function_count(), 574);
    assert_eq!(r.version(), 0);
    let names: Vec<&str> = r.functions().iter().map(|d| d.name).collect();
    assert!(names.windows(2).all(|w| w[0] < w[1]));
    for d in r.functions() {
        let s = r.spark_function(d.name).unwrap();
        assert_eq!(d.since, s.since, "{}", d.name);
        assert_eq!(Some(d.deterministic), s.deterministic, "{}", d.name);
        assert_eq!(s.kind, Some(d.kind), "{}", d.name);
        let dbg = format!("{d:?}");
        assert!(dbg.contains(d.name), "{dbg}");
    }
}

#[test]
fn every_scalar_definition_binds_and_every_binding_is_defined() {
    let defined: Vec<&str> = builtins()
        .functions()
        .iter()
        .filter(|d| d.execution == Execution::Scalar)
        .map(|d| d.name)
        .collect();
    assert_eq!(defined, scalar_names());
}

#[test]
fn lookups_are_case_insensitive_and_versioned() {
    let r = builtins();
    assert!(matches!(r.lookup("UpPeR", SparkProfile::V3_5), Lookup::Found(d) if d.name == "upper"));
    // `randstr` arrived in 4.0 and is not implemented.
    assert!(matches!(
        r.lookup("randstr", SparkProfile::V3_5),
        Lookup::NotFound
    ));
    assert!(matches!(
        r.lookup("RANDSTR", SparkProfile::V4_0),
        Lookup::KnownUnsupported
    ));
    assert!(matches!(
        r.lookup("no_such_fn", SparkProfile::V4_2),
        Lookup::NotFound
    ));
    // `curdate` arrived in 3.4/3.5's successor lines: check its catalog line.
    let since = r.spark_function("curdate").unwrap().since;
    for p in SparkProfile::ALL {
        let found = matches!(r.lookup("curdate", p), Lookup::Found(_));
        assert_eq!(found, p >= since, "{p:?}");
    }
}

#[test]
fn kinds_determinism_and_execution() {
    for (name, agg) in [
        ("count", Aggregate::Count),
        ("sum", Aggregate::Sum),
        ("avg", Aggregate::Avg),
        ("mean", Aggregate::Avg),
        ("min", Aggregate::Min),
        ("max", Aggregate::Max),
    ] {
        let d = def(name);
        assert_eq!(d.kind, FunctionKind::Aggregate);
        assert_eq!(d.execution, Execution::Aggregate(agg));
        assert!(d.deterministic);
    }
    for name in [
        "if",
        "coalesce",
        "ifnull",
        "nvl",
        "nvl2",
        "nullif",
        "nullifzero",
        "zeroifnull",
        "nanvl",
    ] {
        let d = def(name);
        assert_eq!(d.execution, Execution::Evaluator, "{name}");
        assert_eq!(d.kind, FunctionKind::Scalar, "{name}");
    }
    for name in ["rand", "randn", "uuid", "shuffle"] {
        // Not implemented here, but the optimizer's flag comes from Spark.
        assert_eq!(
            builtins().spark_function(name).unwrap().deterministic,
            Some(false),
            "{name}"
        );
    }
    for name in [
        "round", "pmod", "hash", "try_add", "greatest", "abs", "sha2",
    ] {
        assert!(def(name).deterministic, "{name}");
        assert_eq!(def(name).execution, Execution::Scalar, "{name}");
    }
    assert!(def("now").deterministic);
}

#[test]
fn arity_comes_from_spark() {
    let sub = &def("substring").arity;
    assert!(!sub.accepts(1) && sub.accepts(2) && sub.accepts(3) && !sub.accepts(4));
    assert_eq!(sub.expected(), "[2, 3]");
    assert_eq!(def("count").arity.expected(), " >= 1");
    assert!(def("current_date").arity.accepts(0));
    assert!(!def("current_date").arity.accepts(1));
}

#[test]
fn string_signatures() {
    let built =
        |name: &str, args: &[ArgType]| def(name).inputs.expected(&CallTypes::new(args, &[]));
    let (s, b, i) = (
        ty(DataType::String),
        ty(DataType::Binary),
        ty(DataType::Integer),
    );
    let exact = InputType::Exact;
    assert_eq!(
        built("startswith", &[b.clone(), b.clone()]),
        Some(vec![exact(DataType::Binary)])
    );
    assert_eq!(
        built("contains", &[b.clone(), s.clone()]),
        Some(vec![exact(DataType::String)])
    );
    let pad = |t: DataType| Some(vec![exact(t.clone()), exact(DataType::Integer), exact(t)]);
    assert_eq!(
        built("lpad", &[b.clone(), i.clone()]),
        pad(DataType::Binary)
    );
    assert_eq!(
        built("rpad", &[b.clone(), i.clone(), b.clone()]),
        pad(DataType::Binary)
    );
    assert_eq!(
        built("rpad", &[b.clone(), i.clone(), s.clone()]),
        pad(DataType::String)
    );
    assert_eq!(
        built("lpad", &[s.clone(), i.clone()]),
        pad(DataType::String)
    );
    assert_eq!(
        def("length").inputs,
        Inputs::Expects(vec![InputType::OneOf(vec![
            DataType::String,
            DataType::Binary
        ])])
    );
    let r = result("substr", &[b.clone(), i.clone()], &[]).unwrap();
    assert_eq!(r, ty(DataType::Binary));
    let r = result(
        "lpad",
        &[ArgType::new(DataType::String, true), i.clone()],
        &[],
    )
    .unwrap();
    assert_eq!(r, ArgType::new(DataType::String, true));
    assert_eq!(result("upper", std::slice::from_ref(&s), &[]).unwrap(), s);
    assert_eq!(result("length", &[b], &[]).unwrap(), i);
    assert_eq!(
        result("contains", &[s.clone(), s], &[]).unwrap(),
        ty(DataType::Boolean)
    );
    assert_eq!(
        result("substring", &[null_ty(), i], &[]).unwrap(),
        ArgType::new(DataType::String, true)
    );
}

#[test]
fn datetime_signatures() {
    let d = ty(DataType::Date);
    assert_eq!(
        result("date_add", &[d.clone(), ty(DataType::Short)], &[]).unwrap(),
        d
    );
    assert_eq!(
        result("datediff", &[d.clone(), d.clone()], &[]).unwrap(),
        ty(DataType::Integer)
    );
    assert_eq!(
        result("year", std::slice::from_ref(&d), &[]).unwrap(),
        ty(DataType::Integer)
    );
    assert_eq!(result("now", &[], &[]).unwrap(), ty(DataType::Timestamp));
    assert_eq!(result("curdate", &[], &[]).unwrap(), d);
    let field = |s: &str| Some(Constant::String(s.to_owned()));
    let args = [
        ty(DataType::String),
        ArgType::new(DataType::Timestamp, true),
    ];
    let r = result("extract", &args, &[field("SECONDS")]).unwrap();
    assert_eq!(r, ArgType::new(dec(8, 6), true));
    let r = result("date_part", &args, &[field("dow")]).unwrap();
    assert_eq!(r, ArgType::new(DataType::Integer, true));
    let r = result("date_part", &args, &[Some(Constant::Null)]).unwrap();
    assert_eq!(r, ArgType::new(DataType::Double, true));
    let sql = |i: usize| (i == 1).then(|| "d".to_owned());
    let consts = [field("fortnight")];
    let err = (def("extract").result)(&CallTypes::new(&args, &consts).with_sql(&sql)).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidExtractField);
    assert!(
        err.to_string()
            .contains("Cannot extract `fortnight` from \"d\""),
        "{err}"
    );
    let err = result("date_part", &args, &[]).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::NonFoldableArgument);
    let err = result("date_part", &args, &[Some(Constant::Integral(1))]).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::NonFoldableArgument);
    // The source's cast follows the field.
    let inputs = |c: Option<Constant>| def("extract").inputs.expected(&CallTypes::new(&args, &[c]));
    let ts = InputType::OneOf(vec![DataType::Timestamp, DataType::TimestampNtz]);
    assert_eq!(inputs(field("hour")).unwrap()[1], ts);
    assert_eq!(
        inputs(field("year")).unwrap()[1],
        InputType::Exact(DataType::Date)
    );
    assert_eq!(inputs(field("nope")).unwrap()[1], InputType::Any);
    assert_eq!(inputs(Some(Constant::Null)).unwrap()[1], InputType::Any);
}

#[test]
fn aggregate_result_types() {
    let null = |t: DataType| ArgType::new(t, true);
    assert_eq!(
        result("sum", &[ty(DataType::Integer)], &[]).unwrap(),
        null(DataType::Long)
    );
    assert_eq!(
        result("sum", &[ty(DataType::Float)], &[]).unwrap(),
        null(DataType::Double)
    );
    assert_eq!(
        result("sum", &[null_ty()], &[]).unwrap(),
        null(DataType::Double)
    );
    assert_eq!(
        result("sum", &[ty(dec(30, 2))], &[]).unwrap(),
        null(dec(38, 2))
    );
    assert_eq!(
        result("sum", &[ty(dec(10, 2))], &[]).unwrap(),
        null(dec(20, 2))
    );
    assert_eq!(
        result("avg", &[ty(dec(36, 2))], &[]).unwrap(),
        null(dec(38, 6))
    );
    assert_eq!(
        result("mean", &[ty(DataType::Long)], &[]).unwrap(),
        null(DataType::Double)
    );
    assert_eq!(
        result("count", &[null_ty()], &[]).unwrap(),
        ty(DataType::Long)
    );
    assert_eq!(
        result("max", &[ty(DataType::String)], &[]).unwrap(),
        null(DataType::String)
    );
    let r = result("coalesce", &[null_ty(), ty(DataType::Integer)], &[]).unwrap();
    assert_eq!(r, ty(DataType::Integer));
}

#[test]
fn legacy_day_counts_and_binary_predicates() {
    let args = [ty(DataType::Date), ty(DataType::String)];
    let s = |v: &str| [Some(Constant::String(v.to_owned()))];
    let legacy = |name: &str, v: &str| {
        let consts = [None, s(v)[0].clone()];
        (def(name).result)(&CallTypes::new(&args, &consts).with_ansi(false))
    };
    for ok in ["1", " -2 ", "+3", "1.9", "7."] {
        assert!(legacy("date_add", ok).is_ok(), "{ok}");
    }
    for (name, shown) in [
        ("date_add", "date_add"),
        ("dateadd", "date_add"),
        ("date_sub", "date_sub"),
    ] {
        let err = legacy(name, "abc").unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::SecondFunctionArgumentNotInteger
        );
        assert!(
            err.to_string().contains(&format!("of {shown} function")),
            "{err}"
        );
    }
    for bad in ["", ".5", "1e3", "99999999999", "1-"] {
        assert!(legacy("date_sub", bad).is_err(), "{bad}");
    }
    // ANSI on: the cast fails at run time instead.
    let consts = [None, s("abc")[0].clone()];
    assert!((def("date_add").result)(&CallTypes::new(&args, &consts)).is_ok());
    let b = ty(DataType::Binary);
    let r = result("startswith", &[b.clone(), b], &[]).unwrap();
    assert_eq!(r, ArgType::new(DataType::Boolean, true));
}

fn nullable(t: DataType) -> ArgType {
    ArgType::new(t, true)
}

#[test]
fn math_signatures() {
    let (d, i, l) = (
        ty(DataType::Double),
        ty(DataType::Integer),
        ty(DataType::Long),
    );
    // `double` functions are NULL outside their domain; `pow` only for NULLs.
    assert_eq!(
        result("sqrt", std::slice::from_ref(&d), &[]).unwrap(),
        nullable(DataType::Double)
    );
    assert_eq!(result("pow", &[d.clone(), d.clone()], &[]).unwrap(), d);
    assert_eq!(
        result("hypot", &[d.clone(), nullable(DataType::Double)], &[]).unwrap(),
        nullable(DataType::Double)
    );
    assert_eq!(result("pi", &[], &[]).unwrap(), d);
    assert_eq!(
        def("sqrt").inputs,
        Inputs::Expects(vec![InputType::Exact(DataType::Double)])
    );
    // `round`: a decimal follows the (folded) scale, anything else keeps
    // its type; always nullable.
    let scale = |n: i64| [None, Some(Constant::Integral(n))];
    let r = result("round", &[ty(dec(2, 1)), i.clone()], &scale(0)).unwrap();
    assert_eq!(r, nullable(dec(2, 0)));
    let r = result("bround", &[ty(dec(5, 4)), i.clone()], &scale(2)).unwrap();
    assert_eq!(r, nullable(dec(4, 2)));
    let r = result("round", &[i.clone(), i.clone()], &scale(-1)).unwrap();
    assert_eq!(r, nullable(DataType::Integer));
    let r = result(
        "round",
        &[ty(dec(2, 1)), i.clone()],
        &[None, Some(Constant::Fractional(1))],
    );
    assert_eq!(r.unwrap(), nullable(dec(3, 1)));
    let err = result("round", &[ty(dec(2, 1)), i.clone()], &[]).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::NonFoldableArgument);
    // `ceil`/`floor`: integral and floating input give `bigint`; with a scale
    // only an `int` literal is folded.
    assert_eq!(
        result("ceil", std::slice::from_ref(&l), &[]).unwrap(),
        nullable(DataType::Long)
    );
    let r = result("floor", &[ty(dec(5, 4)), i.clone()], &scale(3)).unwrap();
    assert_eq!(r, nullable(dec(5, 3)));
    let err = result("ceil", &[ty(dec(5, 4)), l], &scale(3)).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::NonFoldableArgument);
    let built =
        |name: &str, args: &[ArgType]| def(name).inputs.expected(&CallTypes::new(args, &[]));
    assert_eq!(
        built("ceil", &[ty(DataType::Byte)]),
        Some(vec![InputType::Exact(DataType::Long), InputType::Any])
    );
    assert_eq!(
        built("ceil", &[ty(DataType::Double)]),
        Some(vec![InputType::Exact(DataType::Double), InputType::Any])
    );
    // `greatest` is NULL only when every argument may be; `abs` keeps its
    // argument's type and nullability.
    let r = result("greatest", &[i.clone(), nullable(DataType::Integer)], &[]).unwrap();
    assert_eq!(r, i);
    assert_eq!(def("greatest").inputs, Inputs::CommonNoStrings);
    assert_eq!(
        result("abs", &[nullable(DataType::Short)], &[]).unwrap(),
        nullable(DataType::Short)
    );
    assert_eq!(
        result("abs", &[null_ty()], &[]).unwrap(),
        nullable(DataType::Double)
    );
}

#[test]
fn arithmetic_signatures_follow_the_operators() {
    let (i, l) = (ty(DataType::Integer), ty(DataType::Long));
    // Always nullable: overflow and zero divisors are NULL.
    assert_eq!(
        result("try_add", &[l.clone(), l], &[]).unwrap(),
        nullable(DataType::Long)
    );
    assert_eq!(
        result("mod", &[i.clone(), i.clone()], &[]).unwrap(),
        nullable(DataType::Integer)
    );
    assert_eq!(
        result("pmod", &[i.clone(), i], &[]).unwrap(),
        nullable(DataType::Integer)
    );
    assert_eq!(
        result(
            "try_divide",
            &[ty(DataType::Double), ty(DataType::Double)],
            &[]
        )
        .unwrap(),
        nullable(DataType::Double)
    );
    // Decimals take the operator's precision rules, and the setting.
    let args = [ty(dec(10, 2)), ty(dec(10, 2))];
    assert_eq!(result("try_add", &args, &[]).unwrap(), nullable(dec(11, 2)));
    assert_eq!(
        result("try_multiply", &args, &[]).unwrap(),
        nullable(dec(21, 4))
    );
    let divide = |args: &[ArgType], loss: bool| {
        (def("try_divide").result)(&CallTypes::new(args, &[]).with_decimal_loss(loss)).unwrap()
    };
    assert_eq!(divide(&args, true), nullable(dec(23, 13)));
    // Past 38 digits the setting decides how the scale is cut.
    let wide = [ty(dec(30, 10)), ty(dec(30, 10))];
    assert_ne!(divide(&wide, true), divide(&wide, false));
    assert_eq!(def("try_add").inputs, Inputs::TryOperator(Operator::Add));
    assert_eq!(def("mod").inputs, Inputs::Operator(Operator::Remainder));
}

#[test]
fn hash_signatures() {
    let (s, b) = (ty(DataType::String), ty(DataType::Binary));
    assert_eq!(
        result("hash", &[null_ty(), s.clone()], &[]).unwrap(),
        ty(DataType::Integer)
    );
    assert_eq!(
        result("xxhash64", std::slice::from_ref(&s), &[]).unwrap(),
        ty(DataType::Long)
    );
    assert_eq!(def("hash").inputs, Inputs::Any);
    // The digests take `binary`; NULL only for NULL input (`sha2` also for
    // an unknown length).
    assert_eq!(
        def("md5").inputs,
        Inputs::Expects(vec![InputType::Exact(DataType::Binary)])
    );
    assert_eq!(
        result("md5", std::slice::from_ref(&b), &[]).unwrap(),
        ty(DataType::String)
    );
    assert_eq!(
        result("sha1", &[nullable(DataType::Binary)], &[]).unwrap(),
        nullable(DataType::String)
    );
    assert_eq!(
        result("crc32", std::slice::from_ref(&b), &[]).unwrap(),
        ty(DataType::Long)
    );
    assert_eq!(
        result("sha2", &[b, ty(DataType::Integer)], &[]).unwrap(),
        nullable(DataType::String)
    );
}

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
