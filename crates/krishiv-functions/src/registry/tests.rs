// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use krishiv_common::error::{ErrorClass, InvalidOptionsNonMapFunction, KrishivError};
use krishiv_types::{ArrayType, DataType, DecimalType};

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
        // Spark rejected every probe of `raise_error` (it always raises),
        // so the catalog knows neither its determinism nor its kind; a
        // raising function is fixed as nondeterministic so the optimizer
        // never moves it.
        if d.name == "raise_error" {
            assert!(s.deterministic.is_none() && s.kind.is_none());
            assert!(!d.deterministic);
        } else {
            assert_eq!(Some(d.deterministic), s.deterministic, "{}", d.name);
            assert_eq!(s.kind, Some(d.kind), "{}", d.name);
        }
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
    // `randstr` arrived in 4.0 and is implemented.
    assert!(matches!(
        r.lookup("randstr", SparkProfile::V3_5),
        Lookup::NotFound
    ));
    assert!(matches!(
        r.lookup("RANDSTR", SparkProfile::V4_0),
        Lookup::Found(_)
    ));
    // So are `uniform` (4.0) and the misc functions.
    for name in [
        "uniform",
        "uuid",
        "typeof",
        "version",
        "current_database",
        "current_catalog",
        "monotonically_increasing_id",
        "spark_partition_id",
        "input_file_name",
    ] {
        assert!(
            matches!(r.lookup(name, SparkProfile::V4_2), Lookup::Found(_)),
            "{name}"
        );
    }
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
    for name in ["rand", "randn", "uuid", "uniform", "randstr", "shuffle"] {
        // The optimizer's flag comes from Spark; the implemented ones are
        // nondeterministic scalar bindings (`shuffle` stays unimplemented:
        // it needs an array type the engine has no representation for).
        assert_eq!(
            builtins().spark_function(name).unwrap().deterministic,
            Some(false),
            "{name}"
        );
    }
    for name in ["rand", "randn", "uuid", "uniform", "randstr"] {
        let d = def(name);
        assert!(!d.deterministic, "{name}");
        assert_eq!(d.execution, Execution::Scalar, "{name}");
        assert_eq!(d.kind, FunctionKind::Scalar, "{name}");
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
fn calendar_signatures() {
    let date = ty(DataType::Date);
    let maybe = |t: DataType| ArgType::new(t, true);
    let inputs = |name: &str, args: &[ArgType]| {
        def(name)
            .inputs
            .expected(&CallTypes::new(args, &[]))
            .unwrap()
    };
    // The accessors read a date and give an `int`, as nullable as the date.
    for name in ["dayofweek", "dayofyear", "weekofyear", "weekday", "quarter"] {
        assert_eq!(
            result(name, std::slice::from_ref(&date), &[]).unwrap(),
            ty(DataType::Integer)
        );
        assert_eq!(
            result(name, &[maybe(DataType::Date)], &[]).unwrap(),
            maybe(DataType::Integer)
        );
        assert_eq!(
            inputs(name, std::slice::from_ref(&date)),
            [InputType::Exact(DataType::Date)]
        );
    }
    // The time fields read either timestamp; a date is cast to the first.
    let any_ts = InputType::OneOf(vec![DataType::Timestamp, DataType::TimestampNtz]);
    for name in ["hour", "minute", "second"] {
        assert_eq!(
            inputs(name, std::slice::from_ref(&date)),
            std::slice::from_ref(&any_ts)
        );
        assert_eq!(
            result(name, &[ty(DataType::TimestampNtz)], &[]).unwrap(),
            ty(DataType::Integer)
        );
    }
    for name in ["monthname", "dayname"] {
        assert_eq!(
            result(name, std::slice::from_ref(&date), &[]).unwrap(),
            ty(DataType::String)
        );
        assert_eq!(
            result(name, &[maybe(DataType::Date)], &[]).unwrap(),
            maybe(DataType::String)
        );
    }
    // `unix_date` casts nothing to a date.
    assert_eq!(
        inputs("unix_date", std::slice::from_ref(&date)),
        [InputType::Strict(DataType::Date)]
    );
    assert_eq!(
        result("unix_date", std::slice::from_ref(&date), &[]).unwrap(),
        ty(DataType::Integer)
    );
    assert_eq!(
        inputs("add_months", &[date.clone(), ty(DataType::Integer)]),
        [
            InputType::Exact(DataType::Date),
            InputType::Exact(DataType::Integer)
        ]
    );
    assert_eq!(
        result("add_months", &[date.clone(), ty(DataType::Integer)], &[]).unwrap(),
        date
    );
    assert_eq!(
        result("last_day", std::slice::from_ref(&date), &[]).unwrap(),
        date
    );
    // An unknown weekday or unit is NULL, so these are nullable whatever
    // their arguments.
    let text = ty(DataType::String);
    for name in ["next_day", "trunc"] {
        assert_eq!(
            result(name, &[date.clone(), text.clone()], &[]).unwrap(),
            maybe(DataType::Date)
        );
    }
    assert_eq!(
        result("date_trunc", &[text.clone(), ty(DataType::Timestamp)], &[]).unwrap(),
        maybe(DataType::Timestamp)
    );
    // `date_trunc` and `months_between` keep a `TIMESTAMP_NTZ`; every other
    // type is cast to (or refused as) `TIMESTAMP`.
    let ntz = InputType::OneOf(vec![DataType::Timestamp, DataType::TimestampNtz]);
    let ts = InputType::Exact(DataType::Timestamp);
    assert_eq!(
        inputs("date_trunc", &[text.clone(), ty(DataType::TimestampNtz)])[1],
        ntz
    );
    assert_eq!(
        inputs("date_trunc", &[text.clone(), ty(DataType::Date)])[1],
        ts
    );
    let three = inputs(
        "months_between",
        &[ty(DataType::TimestampNtz), ty(DataType::Date)],
    );
    assert_eq!(
        three,
        [ntz.clone(), ts, InputType::Exact(DataType::Boolean)]
    );
    assert_eq!(
        result(
            "months_between",
            &[ty(DataType::Timestamp), ty(DataType::Timestamp)],
            &[]
        )
        .unwrap(),
        ty(DataType::Double)
    );
    assert_eq!(
        result(
            "months_between",
            &[ty(DataType::Timestamp), maybe(DataType::Timestamp)],
            &[]
        )
        .unwrap(),
        maybe(DataType::Double)
    );
    // `make_date` is NULL for an invalid date unless ANSI raises.
    let ints = [
        ty(DataType::Integer),
        ty(DataType::Integer),
        ty(DataType::Integer),
    ];
    let types = |ansi: bool| {
        (def("make_date").result)(&CallTypes::new(&ints, &[]).with_ansi(ansi)).unwrap()
    };
    assert_eq!(types(true), date);
    assert_eq!(types(false), maybe(DataType::Date));
    assert_eq!(
        inputs("make_date", &ints),
        [InputType::Exact(DataType::Integer)]
    );
    // Zone conversions.
    assert_eq!(
        result(
            "from_utc_timestamp",
            &[ty(DataType::Timestamp), text.clone()],
            &[]
        )
        .unwrap(),
        ty(DataType::Timestamp)
    );
    assert_eq!(
        inputs(
            "to_utc_timestamp",
            &[ty(DataType::TimestampNtz), text.clone()]
        ),
        [ntz, InputType::Exact(DataType::String)]
    );
    let wall = ty(DataType::TimestampNtz);
    assert_eq!(
        result(
            "convert_timezone",
            &[text.clone(), text.clone(), wall.clone()],
            &[]
        )
        .unwrap(),
        ty(DataType::TimestampNtz)
    );
    assert_eq!(
        inputs("convert_timezone", &[text.clone(), wall]),
        [
            InputType::Exact(DataType::String),
            InputType::Exact(DataType::TimestampNtz)
        ]
    );
    assert_eq!(
        inputs(
            "convert_timezone",
            &[text.clone(), text, ty(DataType::Timestamp)]
        )[2],
        InputType::OneOf(vec![DataType::TimestampNtz, DataType::Timestamp])
    );
    // Availability and determinism come from Spark: `monthname` and `dayname`
    // arrived in 4.0.
    for name in ["monthname", "dayname"] {
        assert!(matches!(
            builtins().lookup(name, SparkProfile::V3_5),
            Lookup::NotFound
        ));
        assert!(matches!(
            builtins().lookup(name, SparkProfile::V4_0),
            Lookup::Found(_)
        ));
    }
    for name in [
        "hour",
        "minute",
        "second",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "weekday",
        "quarter",
        "monthname",
        "dayname",
        "unix_date",
        "add_months",
        "last_day",
        "next_day",
        "trunc",
        "date_trunc",
        "months_between",
        "make_date",
        "from_utc_timestamp",
        "to_utc_timestamp",
        "convert_timezone",
    ] {
        assert!(def(name).deterministic, "{name}");
        assert_eq!(def(name).execution, Execution::Scalar, "{name}");
    }
    // What stays unregistered is known to Spark, so it is not "not found".
    for name in [
        "make_timestamp",
        "try_make_timestamp",
        "make_timestamp_ltz",
        "make_timestamp_ntz",
        "make_interval",
    ] {
        assert!(
            matches!(
                builtins().lookup(name, SparkProfile::V4_2),
                Lookup::KnownUnsupported
            ),
            "{name}"
        );
    }
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
    for ok in ["1", " -2 ", "+3", "-0", "2147483647", "007"] {
        assert!(legacy("date_add", ok).is_ok(), "{ok}");
    }
    // The day count the analyzer shows is read by the same function.
    for (text, days) in [(" 12 ", Some(12)), ("-2", Some(-2)), ("+3", Some(3))] {
        assert_eq!(super::legacy_day_count(text), days, "{text:?}");
    }
    for bad in ["1.5", "1.", "1e2", "0x1", "2147483648", "abc", ""] {
        assert_eq!(super::legacy_day_count(bad), None, "{bad:?}");
        assert!(legacy("date_add", bad).is_err(), "{bad:?}");
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
    // A fraction is not an integer to `date_add` (Spark 3.5 to 4.2 raise on
    // `'1.9'`, checked on each), nor are an exponent, a hex number or a
    // value past `int`. Spark rejects a fractional string even with ANSI
    // off: the day count must parse as an `int`, not truncate.
    for bad in [
        "",
        ".5",
        "1e3",
        "99999999999",
        "1-",
        "1.2",
        "1.9",
        "7.",
        "1.0",
        "0x1",
        "2147483648",
        "1 2",
    ] {
        let err = legacy("date_add", bad).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::SecondFunctionArgumentNotInteger,
            "{bad}"
        );
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

/// Spark 4.2 (`df.schema` over `spark.range(1)`, a non-nullable column):
/// every unary math function is nullable, `pow`/`atan2`/`hypot` are
/// nullable only with their inputs, `log(base, x)` is always nullable.
#[test]
fn unary_math_nullability_matches_spark() {
    let d = ty(DataType::Double);
    for name in [
        "sqrt", "cbrt", "exp", "expm1", "ln", "log10", "log2", "log1p", "sin", "cos", "tan",
        "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "cot", "csc",
        "sec", "degrees", "radians", "rint", "sign", "signum", "log",
    ] {
        let r = result(name, std::slice::from_ref(&d), &[]).unwrap();
        assert_eq!(r, nullable(DataType::Double), "{name}");
    }
    for name in ["pow", "power", "atan2", "hypot"] {
        assert_eq!(
            result(name, &[d.clone(), d.clone()], &[]).unwrap(),
            d,
            "{name}"
        );
    }
    assert_eq!(result("pi", &[], &[]).unwrap(), d);
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
    // `round`/`bround` check the scale as an input (whatever the type of the
    // value); `ceil`/`floor` raise NON_FOLDABLE_ARGUMENT.
    let err = result("round", &[ty(dec(2, 1)), i.clone()], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    let err = result("bround", &[ty(DataType::Double), i.clone()], &[]).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    // A cast of a literal that failed while folding is raised by the reader.
    let failed = Constant::Error(KrishivError::from(krishiv_common::error::CastOverflow {
        value: "1.0E30D".to_owned(),
        source_type: "\"DOUBLE\"".to_owned(),
        target_type: "\"INT\"".to_owned(),
    }));
    let err = result("round", &[ty(dec(2, 1)), i.clone()], &[None, Some(failed)]).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::CastOverflow);
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
    // The aggregates that return arrays stay known-unsupported.
    for name in ["collect_list", "collect_set", "array_agg"] {
        assert!(
            matches!(
                builtins().lookup(name, SparkProfile::V4_2),
                Lookup::KnownUnsupported
            ),
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
    // `current_time` is a `TIME`, which Krishiv has no type for: it stays
    // known-unsupported where Spark has it (4.1+).
    assert!(matches!(
        builtins().lookup("current_time", SparkProfile::V4_2),
        Lookup::KnownUnsupported
    ));
    // `shuffle` needs an array type the engine has no representation for:
    // it stays known-unsupported where Spark has it.
    assert!(matches!(
        builtins().lookup("shuffle", SparkProfile::V4_2),
        Lookup::KnownUnsupported
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

/// The analyzer's `uniform` result type and the executor's bind type agree
/// on every pair of bound kinds: both layers map their own type to the
/// shared `uniform_result` rule, and the test pins the two mappers against
/// each other so one side cannot drift without the other.
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
                    a.clone().unwrap_or(DataType::Null).to_arrow().into(),
                    b.clone().unwrap_or(DataType::Null).to_arrow().into(),
                ],
                &BindOptions::new(),
            )
            .unwrap();
            assert_eq!(
                reg.data_type.clone().map(|t| t.to_arrow()),
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
    // `json_array_length` and `get_json_object` take strings exactly.
    assert_eq!(
        def("json_array_length").inputs,
        Inputs::Expects(vec![InputType::Exact(DataType::String)])
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
    for name in ["transform", "filter", "exists", "forall"] {
        assert_eq!(def(name).execution, Execution::HigherOrder, "{name}");
        assert_eq!(def(name).kind, FunctionKind::Scalar, "{name}");
    }
    assert_eq!(def("array").execution, Execution::Scalar);
    // The arity check runs before typing, so a call typed without its
    // array argument is an engine bug, never a user error.
    for name in ["transform", "filter", "exists", "forall"] {
        let err = result(name, &[], &[]).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::InternalError, "{name}");
    }
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

mod text;

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
