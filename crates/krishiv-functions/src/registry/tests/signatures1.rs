// SPDX-License-Identifier: Apache-2.0
//! Signature tests, part of the registry tests.

use super::*;

#[test]
fn every_definition_is_in_sparks_catalog() {
    let r = builtins();
    assert!(r.unlisted().is_empty(), "{:?}", r.unlisted());
    assert_eq!(r.spark_function_count(), 574);
    assert_eq!(r.version(), 0);
    let names: Vec<&str> = r.functions().iter().map(|d| d.name).collect();
    assert!(names.windows(2).all(|w| w[0] < w[1]));
    for d in r.functions() {
        if is_grammar_only(d.name) {
            assert!(r.spark_function(d.name).is_none(), "{}", d.name);
            continue;
        }
        let s = r.spark_function(d.name).unwrap();
        assert_eq!(d.since, s.since, "{}", d.name);
        // Spark rejected every probe of `raise_error` (it always raises),
        // so the catalog knows neither its determinism nor its kind; a
        // raising function is fixed as nondeterministic so the optimizer
        // never moves it.
        if d.name == "raise_error" {
            assert!(s.deterministic.is_none() && s.kind.is_none());
            assert!(!d.deterministic);
        } else if matches!(d.name, "from_avro" | "to_avro") {
            // The `spark-avro` module is not on the probe image's classpath, so
            // Spark rejected every probe of these two: the catalog has their
            // arity and first release but not their determinism or kind (so the
            // registry treats them as non-deterministic, which only costs folding).
            assert!(s.deterministic.is_none() && s.kind.is_none(), "{}", d.name);
            assert!(!d.deterministic, "{}", d.name);
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
        // nondeterministic scalar bindings.
        assert_eq!(
            builtins().spark_function(name).unwrap().deterministic,
            Some(false),
            "{name}"
        );
    }
    for name in ["rand", "randn", "uuid", "uniform", "randstr", "shuffle"] {
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
    let ts = timestamp_or_time();
    assert_eq!(inputs(field("hour")).unwrap()[1], ts);
    assert_eq!(
        inputs(field("year")).unwrap()[1],
        InputType::Exact(DataType::Date)
    );
    assert_eq!(inputs(field("nope")).unwrap()[1], InputType::Any);
    assert_eq!(inputs(Some(Constant::Null)).unwrap()[1], InputType::Any);
}

#[test]
fn pattern_signatures() {
    // Pinned against Spark 4.2 runs (the function matrix): the result
    // types and nullability of the pattern-driven datetime functions.
    let maybe = |t: DataType| ArgType::new(t, true);
    let inputs = |name: &str, args: &[ArgType]| {
        def(name)
            .inputs
            .expected(&CallTypes::new(args, &[]))
            .unwrap()
    };
    let string = || ty(DataType::String);
    let date = || ty(DataType::Date);
    let collection = || {
        InputType::OneOf(vec![
            DataType::String,
            DataType::Date,
            DataType::Timestamp,
            DataType::TimestampNtz,
        ])
    };
    // `date_format` reads a date-like value and a pattern.
    assert_eq!(
        inputs("date_format", &[string(), string()]),
        [
            InputType::StrictOneOf(vec![
                DataType::String,
                DataType::Date,
                DataType::Timestamp,
                DataType::TimestampNtz,
            ]),
            InputType::Exact(DataType::String)
        ]
    );
    assert_eq!(
        result("date_format", &[string(), string()], &[]).unwrap(),
        maybe(DataType::String)
    );
    assert_eq!(
        result("date_format", &[date(), string()], &[]).unwrap(),
        ty(DataType::String)
    );
    // `to_date`: one argument casts (nullable where the cast may yield
    // NULL); two are `failOnError`-nullable.
    assert_eq!(inputs("to_date", &[string()]), [collection()]);
    assert_eq!(
        result("to_date", &[string()], &[]).unwrap(),
        maybe(DataType::Date)
    );
    assert_eq!(
        result("to_date", &[date()], &[]).unwrap(),
        ty(DataType::Date)
    );
    assert_eq!(
        result("to_date", &[string(), string()], &[]).unwrap(),
        ty(DataType::Date)
    );
    assert_eq!(
        result("try_to_date", &[date()], &[]).unwrap(),
        ty(DataType::Date)
    );
    assert_eq!(
        result("try_to_date", &[string()], &[]).unwrap(),
        maybe(DataType::Date)
    );
    // `to_timestamp`: numerics stay numeric (seconds since the epoch);
    // anything else takes the collection.
    assert_eq!(
        inputs("to_timestamp", &[ty(DataType::Long)]),
        [InputType::Numeric { intervals: false }]
    );
    assert_eq!(inputs("to_timestamp", &[string()]), [collection()]);
    assert_eq!(
        result("to_timestamp", &[ty(DataType::Long)], &[]).unwrap(),
        ty(DataType::Timestamp)
    );
    assert_eq!(
        result("to_timestamp", &[string()], &[]).unwrap(),
        maybe(DataType::Timestamp)
    );
    assert_eq!(
        result("try_to_timestamp", &[ty(DataType::Long)], &[]).unwrap(),
        ty(DataType::Timestamp)
    );
    // `unix_timestamp` takes no numbers and no numeric patterns (the
    // matrix pins the `DATATYPE_MISMATCH` against Spark 4.2 runs).
    assert_eq!(
        inputs("unix_timestamp", &[string(), string()]),
        [
            InputType::StrictOneOf(vec![
                DataType::String,
                DataType::Date,
                DataType::Timestamp,
                DataType::TimestampNtz,
            ]),
            InputType::Strict(DataType::String)
        ]
    );
    // `from_unixtime` reads seconds; `current_timezone` reads nothing.
    assert_eq!(
        result("from_unixtime", &[ty(DataType::Long)], &[]).unwrap(),
        maybe(DataType::String)
    );
    assert_eq!(
        result("current_timezone", &[], &[]).unwrap(),
        ty(DataType::String)
    );
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
    let any_ts = timestamp_or_time();
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
    assert!(
        matches!(
            builtins().lookup("hll_sketch_agg", SparkProfile::V4_2),
            Lookup::KnownUnsupported
        ),
        "hll_sketch_agg"
    );
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

/// The `regr_*`, bitwise, `mode`, `listagg` and `try_*` aggregates: their
/// kinds, determinism, arities, result types and typing errors.
#[test]
fn misc_aggregates() {
    for (name, agg) in [
        ("bit_and", Aggregate::BitAnd),
        ("bit_or", Aggregate::BitOr),
        ("bit_xor", Aggregate::BitXor),
        ("regr_count", Aggregate::RegrCount),
        ("regr_avgx", Aggregate::RegrAvgx),
        ("regr_avgy", Aggregate::RegrAvgy),
        ("regr_sxx", Aggregate::RegrSxx),
        ("regr_syy", Aggregate::RegrSyy),
        ("regr_sxy", Aggregate::RegrSxy),
        ("regr_slope", Aggregate::RegrSlope),
        ("regr_intercept", Aggregate::RegrIntercept),
        ("regr_r2", Aggregate::RegrR2),
        ("mode", Aggregate::Mode),
        ("listagg", Aggregate::ListAgg),
        ("string_agg", Aggregate::ListAgg),
        ("try_sum", Aggregate::TrySum),
        ("try_avg", Aggregate::TryAvg),
    ] {
        let d = def(name);
        assert_eq!(d.kind, FunctionKind::Aggregate, "{name}");
        assert_eq!(d.execution, Execution::Aggregate(agg), "{name}");
        assert!(d.deterministic, "{name}");
    }
    for name in ["bit_and", "bit_or", "bit_xor"] {
        let d = def(name);
        assert!(d.arity.accepts(1) && !d.arity.accepts(2), "{name}");
        for t in [
            DataType::Byte,
            DataType::Short,
            DataType::Integer,
            DataType::Long,
        ] {
            assert_eq!(
                result(name, &[ty(t.clone())], &[]).unwrap(),
                ArgType::new(t, true),
                "{name}"
            );
        }
        let err = result(name, &[ty(DataType::String)], &[]).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{name}"
        );
    }
    // `regr_count` is a non-NULL `bigint`; the rest are nullable
    // `double`s over two `double` arguments.
    assert_eq!(
        result(
            "regr_count",
            &[ty(DataType::Integer), ty(DataType::Integer)],
            &[None, None]
        )
        .unwrap(),
        ArgType::new(DataType::Long, false)
    );
    for name in [
        "regr_avgx",
        "regr_avgy",
        "regr_sxx",
        "regr_syy",
        "regr_sxy",
        "regr_slope",
        "regr_intercept",
        "regr_r2",
    ] {
        let d = def(name);
        assert!(d.arity.accepts(2) && !d.arity.accepts(1), "{name}");
        assert_eq!(
            result(name, &[ty(DataType::Integer), ty(DataType::Integer)], &[]).unwrap(),
            ArgType::new(DataType::Double, true),
            "{name}"
        );
    }
    // `mode` takes any single value and keeps its type.
    assert!(def("mode").arity.accepts(0) && def("mode").arity.accepts(2));
    assert_eq!(
        result("mode", &[ty(DataType::Integer)], &[]).unwrap(),
        ArgType::new(DataType::Integer, true)
    );
    // `listagg` and `string_agg` take a string or binary value with an
    // optional delimiter of the same kind, and keep the value's type.
    // (Both arguments are literals here, as the analyzer sees them.)
    let listagg = |name: &str, args: &[ArgType], literals: &[bool]| {
        let call = CallTypes::new(args, &[None, None]).with_literals(literals);
        (def(name).result)(&call)
    };
    for name in ["listagg", "string_agg"] {
        let d = def(name);
        assert!(d.arity.accepts(1) && d.arity.accepts(2) && !d.arity.accepts(3));
        assert_eq!(
            listagg(
                name,
                &[ty(DataType::String), ty(DataType::Null)],
                &[true, true]
            )
            .unwrap(),
            ArgType::new(DataType::String, true),
            "{name}"
        );
        assert_eq!(
            listagg(
                name,
                &[ty(DataType::Binary), ty(DataType::Binary)],
                &[true, true]
            )
            .unwrap(),
            ArgType::new(DataType::Binary, true),
            "{name}"
        );
        // A string delimiter for binary values is `DATA_DIFF_TYPES`;
        // a non-foldable delimiter is `NON_FOLDABLE_INPUT`.
        let args = [ty(DataType::Binary), ty(DataType::String)];
        let err = listagg(name, &args, &[false, true]).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchDataDiffTypes,
            "{name}"
        );
        let err = listagg(name, &args, &[true, false]).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchNonFoldableInput,
            "{name}"
        );
    }
    // `try_sum` types as `sum`, `try_avg` as `avg`.
    assert_eq!(
        result("try_sum", &[ty(DataType::Integer)], &[]).unwrap(),
        ArgType::new(DataType::Long, true)
    );
    assert_eq!(
        result("try_avg", &[ty(DataType::Integer)], &[]).unwrap(),
        ArgType::new(DataType::Double, true)
    );
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
        assert_eq!(super::super::legacy_day_count(text), days, "{text:?}");
    }
    for bad in ["1.5", "1.", "1e2", "0x1", "2147483648", "abc", ""] {
        assert_eq!(super::super::legacy_day_count(bad), None, "{bad:?}");
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

/// `min`/`max` take any orderable type — structs field by field — and
/// refuse the rest with Spark's `INVALID_ORDERING_TYPE`.
#[test]
fn min_max_signatures_order_structs_field_by_field() {
    use krishiv_types::{MapType, StructField, StructType};

    let struct_of = |t: DataType| {
        ty(DataType::Struct(StructType::new(vec![StructField::new(
            "f", t, true,
        )])))
    };
    let ordered = struct_of(DataType::Integer);
    for name in ["min", "max"] {
        assert_eq!(
            result(name, std::slice::from_ref(&ordered), &[]).unwrap(),
            nullable_struct()
        );
        let nested = struct_of(DataType::Struct(StructType::new(vec![
            StructField::new("x", DataType::Double, true),
            StructField::new("y", DataType::String, true),
        ])));
        assert!(result(name, &[nested], &[]).is_ok());
        // A map field (or one nested in a struct) does not order.
        let map = DataType::Map(MapType::new(DataType::String, DataType::Integer, true).into());
        let nested_map = struct_of(map.clone());
        let nested_map = struct_of(nested_map.data_type.clone().unwrap());
        for bad in [struct_of(map), nested_map] {
            let err = result(name, &[bad], &[]).unwrap_err();
            assert_eq!(
                err.error_class(),
                ErrorClass::DatatypeMismatchInvalidOrderingType,
                "{name}"
            );
        }
        // Spark orders an array field (element by element), and so does the struct comparator.
        let array = DataType::Array(krishiv_types::ArrayType::new(DataType::Integer, true).into());
        assert!(
            result(name, &[struct_of(array.clone())], &[]).is_ok(),
            "{name}"
        );
        // An array of maps does not order.
        let maps = DataType::Array(
            krishiv_types::ArrayType::new(
                DataType::Map(MapType::new(DataType::String, DataType::Integer, true).into()),
                true,
            )
            .into(),
        );
        let err = result(name, &[struct_of(maps)], &[]).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchInvalidOrderingType,
            "{name}"
        );
        // `first`/`last` keep their old rule (any type, no ordering check).
        for other in ["first", "last", "any_value"] {
            assert!(
                result(other, std::slice::from_ref(&ordered), &[]).is_ok(),
                "{other}"
            );
        }
    }
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
