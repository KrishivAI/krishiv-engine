// SPDX-License-Identifier: Apache-2.0
//! Tests of implicit casts, the coercion rules and their errors.

use krishiv_types::DataType as T;

use super::super::catalog::ArgType;
use super::*;

/// `implicit` under ANSI, where a string may reach a BOOLEAN parameter.
fn implicit(from: Option<&DataType>, want: &InputType) -> Option<Option<DataType>> {
    super::implicit(from, want, true)
}

#[test]
fn a_string_reaches_a_boolean_parameter_only_under_ansi() {
    let want = InputType::Exact(T::Boolean);
    assert_eq!(
        super::implicit(Some(&T::String), &want, true),
        Some(Some(T::Boolean))
    );
    assert_eq!(super::implicit(Some(&T::String), &want, false), None);
}

/// ANSI implicit casts of function arguments (`AnsiTypeCoercion
/// .implicitCast`), as Spark 4.2 inserts them, e.g. `date_add('2020-01-01',
/// 1)` casts its string to a date and `upper(dt)` its date to a string.
#[test]
fn implicit_casts_follow_ansi() {
    let exact = |t: T| InputType::Exact(t);
    let numeric = InputType::Numeric { intervals: false };
    assert_eq!(
        implicit(Some(&T::String), &exact(T::Date)),
        Some(Some(T::Date))
    );
    assert_eq!(
        implicit(Some(&T::Date), &exact(T::Timestamp)),
        Some(Some(T::Timestamp))
    );
    assert_eq!(
        implicit(Some(&T::Date), &exact(T::String)),
        Some(Some(T::String))
    );
    assert_eq!(
        implicit(Some(&T::Long), &exact(T::Integer)),
        Some(Some(T::Integer))
    );
    assert_eq!(implicit(None, &exact(T::Integer)), Some(Some(T::Integer)));
    assert_eq!(implicit(Some(&T::Boolean), &exact(T::Integer)), None);
    assert_eq!(implicit(Some(&T::Date), &exact(T::Integer)), None);
    assert_eq!(implicit(Some(&T::String), &numeric), Some(Some(T::Double)));
    assert_eq!(implicit(Some(&T::Boolean), &numeric), None);
    assert_eq!(
        implicit(Some(&T::Binary), &InputType::Any),
        Some(Some(T::Binary))
    );
    assert_eq!(numeric.sql(), r#""NUMERIC""#);
    assert_eq!(
        InputType::Numeric { intervals: true }.sql(),
        r#""NUMERIC" or "ANSI INTERVAL""#
    );
    assert_eq!(InputType::Any.sql(), r#""ANY""#);
    assert_eq!(exact(T::Integer).sql(), r#""INT""#);
    // Spark's `TypeCollection`: members stay, others take the first
    // member they may be cast to, numbers excepted.
    let days = InputType::OneOf(vec![T::Integer, T::Short, T::Byte]);
    assert_eq!(implicit(Some(&T::Byte), &days), Some(Some(T::Byte)));
    assert_eq!(implicit(Some(&T::String), &days), Some(Some(T::Integer)));
    assert_eq!(implicit(None, &days), Some(Some(T::Integer)));
    assert_eq!(implicit(Some(&T::Long), &days), None);
    assert_eq!(implicit(Some(&T::Double), &days), None);
    let text = InputType::OneOf(vec![T::String, T::Binary]);
    assert_eq!(implicit(Some(&T::Binary), &text), Some(Some(T::Binary)));
    assert_eq!(implicit(Some(&T::Boolean), &text), Some(Some(T::String)));
    assert_eq!(implicit(Some(&T::Long), &text), Some(Some(T::String)));
    // `year(ts)` reads the timestamp's date.
    assert_eq!(
        implicit(Some(&T::TimestampNtz), &exact(T::Date)),
        Some(Some(T::Date))
    );
}

/// A canonical operator called with the wrong number of arguments (no
/// front end builds one) is reported, never typed.
#[test]
fn malformed_operator_calls_are_unsupported() {
    let int = Arg {
        ty: Some(&T::Integer),
        nullable: false,
        literal: None,
    };
    assert_eq!(
        operator(Builtin::Not, &[int, int], rules(true), true).err(),
        Some(Mismatch::Unsupported("this operator's arguments"))
    );
    assert_eq!(ordinal(3), "4th");
}

fn rules(ansi: bool) -> Rules {
    Rules {
        ansi,
        case_sensitive: false,
        concat_binary_as_string: false,
    }
}

fn arg(t: Option<&T>) -> Arg<'_> {
    Arg {
        ty: t,
        nullable: false,
        literal: None,
    }
}

fn nullable_arg(t: Option<&T>) -> Arg<'_> {
    Arg {
        ty: t,
        nullable: true,
        literal: None,
    }
}

/// `+` and `-` between a date or timestamp and a DECIMAL fail with the
/// class Spark's `DateTimeOperations` gives each shape.
#[test]
fn datetime_and_decimal_arithmetic_mismatches() {
    let dec = T::Decimal(DecimalType::SYSTEM_DEFAULT);
    let shapes = |op, l: &T, r: &T| datetime_decimal(op, &arg(Some(l)), &arg(Some(r)));
    for ts in [T::Timestamp, T::TimestampNtz] {
        assert!(matches!(
            shapes(Builtin::Add, &ts, &dec),
            Some(Mismatch::DiffTypes(..))
        ));
        assert!(matches!(
            shapes(Builtin::Subtract, &dec, &ts),
            Some(Mismatch::Unexpected {
                arg: 0,
                param: 0,
                ..
            })
        ));
    }
    assert!(matches!(
        shapes(Builtin::Add, &dec, &T::Date),
        Some(Mismatch::DateDays {
            sub: false,
            date: 1,
            ..
        })
    ));
    assert!(matches!(
        shapes(Builtin::Subtract, &T::Date, &dec),
        Some(Mismatch::DateDays {
            sub: true,
            date: 0,
            ..
        })
    ));
    assert!(shapes(Builtin::Add, &T::Date, &T::Integer).is_none());
}

/// `+` between a string and a date or timestamp: a timestamp rejects the
/// number the string is read as, a date reads it as `date_add`'s days.
#[test]
fn datetime_and_string_sums() {
    let text = T::String;
    let sum = |l: &T, r: &T| datetime_text_sum(&arg(Some(l)), &arg(Some(r)), true);
    for ts in [T::Timestamp, T::TimestampNtz] {
        assert_eq!(
            sum(&text, &ts),
            Some(Mismatch::WrongType(Some(ts.clone()), ADDITIVE))
        );
        assert_eq!(
            sum(&ts, &text),
            Some(Mismatch::WrongType(Some(ts.clone()), ADDITIVE))
        );
    }
    assert_eq!(sum(&T::Date, &text), Some(Mismatch::DateAddText));
    assert_eq!(sum(&text, &T::Date), Some(Mismatch::DateAddText));
    assert!(sum(&T::Date, &T::Integer).is_none());
    assert!(sum(&text, &T::Integer).is_none());
    // Without ANSI, Spark's error differs: it stays refused.
    let off = |l: &T, r: &T| datetime_text_sum(&arg(Some(l)), &arg(Some(r)), false);
    assert!(off(&T::Date, &text).is_none());
    assert!(off(&text, &T::Timestamp).is_none());
}

/// Struct comparisons are NULL only when a whole struct operand is NULL:
/// nullable fields never make `=`, `<=>`, `<` and friends nullable.
#[test]
fn struct_comparison_nullability_ignores_fields() {
    let struct_ty = T::Struct(StructType::new([
        StructField::new("a", T::Integer, true),
        StructField::new("b", T::Integer, true),
    ]));
    let non_null = [arg(Some(&struct_ty)), arg(Some(&struct_ty))];
    for op in [
        Builtin::Eq,
        Builtin::Lt,
        Builtin::LtEq,
        Builtin::Gt,
        Builtin::GtEq,
    ] {
        let typed = operator(op, &non_null, rules(true), true).expect("struct comparison");
        assert!(!typed.nullable, "{op:?} with non-null structs");
    }
    // `<=>` is never NULL, even for nullable structs.
    let nullable = [
        nullable_arg(Some(&struct_ty)),
        nullable_arg(Some(&struct_ty)),
    ];
    let typed = operator(Builtin::EqNullSafe, &nullable, rules(true), true)
        .expect("null-safe struct comparison");
    assert!(!typed.nullable);
    // Other comparisons are NULL when a whole struct operand is.
    let typed = operator(Builtin::Eq, &nullable, rules(true), true).expect("nullable struct eq");
    assert!(typed.nullable);
    // A nullable primitive operand still makes the comparison nullable.
    let ints = [arg(Some(&T::Integer)), nullable_arg(Some(&T::Integer))];
    let typed = operator(Builtin::Eq, &ints, rules(true), true).expect("nullable int eq");
    assert!(typed.nullable);
}

/// The casts a registry function's `Inputs` ask for.
fn casts(inputs: &Inputs, args: &[Arg<'_>], ansi: bool) -> Result<Vec<Option<T>>, Mismatch> {
    let types: Vec<ArgType> = args
        .iter()
        .map(|a| ArgType {
            data_type: a.ty.cloned(),
            nullable: a.nullable,
        })
        .collect();
    super::inputs(inputs, args, &CallTypes::new(&types, &[]).with_ansi(ansi))
}

#[test]
fn conditional_inputs_meet_at_a_common_type() {
    let (int, long, string, boolean) = (T::Integer, T::Long, T::String, T::Boolean);
    let leading = Inputs::Leading(vec![InputType::Strict(T::Boolean)]);
    // `if(cond, 1, 2L)`: the branches meet at `bigint`, the condition
    // stays.
    let args = [arg(Some(&boolean)), arg(Some(&int)), arg(Some(&long))];
    assert_eq!(
        casts(&leading, &args, true),
        Ok(vec![None, Some(T::Long), None])
    );
    // An untyped NULL condition is a boolean; a string is no condition.
    assert_eq!(
        casts(
            &leading,
            &[arg(None), arg(Some(&int)), arg(Some(&int))],
            true
        ),
        Ok(vec![Some(T::Boolean), None, None])
    );
    let bad = casts(
        &leading,
        &[arg(Some(&string)), arg(Some(&int)), arg(None)],
        true,
    );
    assert!(
        matches!(bad, Err(Mismatch::Unexpected { arg: 0, .. })),
        "{bad:?}"
    );
    // `coalesce('a', 1)`: ANSI promotes the string to `bigint`, legacy
    // makes both strings; a boolean and a string never meet in legacy.
    let mixed = [arg(Some(&string)), arg(Some(&int))];
    assert_eq!(
        casts(&Inputs::Common, &mixed, true),
        Ok(vec![Some(T::Long), Some(T::Long)])
    );
    assert_eq!(
        casts(&Inputs::Common, &mixed, false),
        Ok(vec![None, Some(T::String)])
    );
    let flag = [arg(Some(&string)), arg(Some(&boolean))];
    assert!(casts(&Inputs::Common, &flag, false).is_err());
    // `greatest('a', 1)` is an error in both modes.
    for ansi in [true, false] {
        assert!(casts(&Inputs::CommonNoStrings, &mixed, ansi).is_err());
    }
    assert!(
        casts(
            &Inputs::CommonNoStrings,
            &[arg(Some(&string)), arg(None)],
            true
        )
        .is_ok()
    );
    // `zeroifnull(x)` meets `x` with `0`: a decimal(10,2) grows to (12,2).
    let dec = T::Decimal(DecimalType::try_new(10, 2).unwrap());
    assert_eq!(
        casts(&Inputs::CommonWithZero, &[arg(Some(&dec))], true),
        Ok(vec![Some(T::Decimal(DecimalType::try_new(12, 2).unwrap()))])
    );
    assert_eq!(
        casts(&Inputs::CommonWithZero, &[arg(None)], true),
        Ok(vec![Some(T::Integer)])
    );
}

#[test]
fn comparable_inputs_are_checked_not_cast() {
    let (int, boolean, string) = (T::Integer, T::Boolean, T::String);
    let eq = Inputs::Comparable { zero: false };
    assert_eq!(
        casts(&eq, &[arg(Some(&int)), arg(Some(&T::Float))], true),
        Ok(vec![None, None])
    );
    let bad = casts(&eq, &[arg(Some(&boolean)), arg(Some(&int))], true);
    assert!(matches!(bad, Err(Mismatch::DiffTypes(..))), "{bad:?}");
    // Without ANSI a boolean compares with a number.
    assert!(casts(&eq, &[arg(Some(&boolean)), arg(Some(&int))], false).is_ok());
    assert!(casts(&eq, &[arg(Some(&string)), arg(Some(&int))], true).is_ok());
    let zero = Inputs::Comparable { zero: true };
    assert!(casts(&zero, &[arg(Some(&int))], true).is_ok());
    assert!(casts(&zero, &[arg(Some(&boolean))], true).is_err());
}

#[test]
fn operator_inputs_cast_as_the_operator_does() {
    let (int, long, float, string) = (T::Integer, T::Long, T::Float, T::String);
    let rem = Inputs::Operator(Operator::Remainder);
    assert_eq!(
        casts(&rem, &[arg(Some(&int)), arg(Some(&long))], true),
        Ok(vec![Some(T::Long), None])
    );
    // `try_divide` of integers divides doubles.
    let div = Inputs::Operator(Operator::Divide);
    assert_eq!(
        casts(&div, &[arg(Some(&int)), arg(Some(&int))], true),
        Ok(vec![Some(T::Double), Some(T::Double)])
    );
    // A `float` meeting an integral type is `double`, or `float` without
    // ANSI.
    let add = Inputs::Operator(Operator::Add);
    let args = [arg(Some(&float)), arg(Some(&int))];
    assert_eq!(
        casts(&add, &args, true),
        Ok(vec![Some(T::Double), Some(T::Double)])
    );
    assert_eq!(casts(&add, &args, false), Ok(vec![None, Some(T::Float)]));
    let bad = casts(&add, &[arg(Some(&T::Boolean)), arg(Some(&string))], true);
    assert!(bad.is_err());
}

#[test]
fn strict_inputs_take_the_exact_type_only() {
    let strict = InputType::Strict(T::Boolean);
    assert_eq!(implicit(Some(&T::Boolean), &strict), Some(Some(T::Boolean)));
    assert_eq!(implicit(None, &strict), Some(Some(T::Boolean)));
    assert_eq!(implicit(Some(&T::String), &strict), None);
    assert_eq!(implicit(Some(&T::Integer), &strict), None);
    assert_eq!(strict.sql(), r#""BOOLEAN""#);
}

/// Nested types meet as Spark 4.2 typed `coalesce` (and `UNION ALL`) of
/// `array(..)`, `map(..)` and `named_struct(..)` values on the oracle,
/// with `spark.sql.ansi.enabled` on and off (types as Spark's JSON).
#[test]
fn nested_types_meet_element_by_element() {
    let t = |json: &str| T::from_json(json).unwrap_or_else(|e| panic!("{json}: {e}"));
    let arr =
        |e: &str, n: bool| format!(r#"{{"type":"array","elementType":{e},"containsNull":{n}}}"#);
    let map = |k: &str, v: &str, n: bool| {
        format!(r#"{{"type":"map","keyType":{k},"valueType":{v},"valueContainsNull":{n}}}"#)
    };
    let st = |fields: &[(&str, &str, bool, &str)]| {
        let fields: Vec<String> = fields
            .iter()
            .map(|(name, ty, n, meta)| {
                format!(r#"{{"name":"{name}","type":{ty},"nullable":{n},"metadata":{meta}}}"#)
            })
            .collect();
        format!(r#"{{"type":"struct","fields":[{}]}}"#, fields.join(","))
    };
    let wider = |a: &str, b: &str, ansi: bool| {
        let (a, b) = (t(a), t(b));
        wider_type(&arg(Some(&a)), &arg(Some(&b)), rules(ansi)).map(|r| r.map(|t| t.json()))
    };
    let int = r#""integer""#;
    let long = r#""long""#;
    let string = r#""string""#;
    let cases: &[(String, String, bool, Option<String>)] = &[
        (
            arr(int, false),
            arr(long, false),
            true,
            Some(arr(long, false)),
        ),
        (arr(int, false), arr(int, true), true, Some(arr(int, true))),
        // A string element cast to bigint may fail, so may be NULL.
        (
            arr(int, false),
            arr(string, false),
            true,
            Some(arr(long, true)),
        ),
        (
            arr(int, false),
            arr(string, false),
            false,
            Some(arr(string, false)),
        ),
        (
            arr(int, false),
            arr(r#""decimal(2,1)""#, false),
            true,
            Some(arr(r#""decimal(11,1)""#, false)),
        ),
        (
            arr(int, false),
            arr(r#""float""#, false),
            true,
            Some(arr(r#""double""#, false)),
        ),
        (
            arr(int, false),
            arr(r#""float""#, false),
            false,
            Some(arr(r#""float""#, false)),
        ),
        (arr(int, false), arr(r#""date""#, false), true, None),
        (
            arr(r#""date""#, false),
            arr(r#""timestamp""#, false),
            true,
            Some(arr(r#""timestamp""#, false)),
        ),
        (
            arr(r#""void""#, true),
            arr(int, false),
            true,
            Some(arr(int, true)),
        ),
        (arr(int, false), int.to_owned(), true, None),
        (
            arr(&arr(int, false), false),
            arr(&arr(long, false), false),
            true,
            Some(arr(&arr(long, false), false)),
        ),
        (
            map(int, string, false),
            map(long, string, false),
            true,
            Some(map(long, string, false)),
        ),
        (
            map(int, string, false),
            map(int, int, false),
            true,
            Some(map(int, long, true)),
        ),
        (
            map(int, string, false),
            map(int, int, false),
            false,
            Some(map(int, string, false)),
        ),
        // A key that may become NULL is no key.
        (map(string, int, false), map(int, int, false), true, None),
        (
            st(&[("a", int, false, "{}")]),
            st(&[("a", long, false, "{}")]),
            true,
            Some(st(&[("a", long, false, "{}")])),
        ),
        (
            st(&[("a", int, false, "{}")]),
            st(&[("A", long, false, "{}")]),
            true,
            Some(st(&[("a", long, false, "{}")])),
        ),
        (
            st(&[("A", int, false, "{}")]),
            st(&[("a", long, false, "{}")]),
            true,
            Some(st(&[("A", long, false, "{}")])),
        ),
        (
            st(&[("a", int, false, "{}")]),
            st(&[("a", int, true, "{}")]),
            true,
            Some(st(&[("a", int, true, "{}")])),
        ),
        (
            st(&[("a", int, false, "{}")]),
            st(&[("b", long, false, "{}")]),
            true,
            None,
        ),
        (
            st(&[("a", int, false, "{}"), ("b", int, false, "{}")]),
            st(&[("a", int, false, "{}")]),
            true,
            None,
        ),
        // Metadata survives only when the types are equal.
        (
            st(&[("a", int, true, r#"{"comment":"x"}"#)]),
            st(&[("a", int, true, r#"{"comment":"y"}"#)]),
            true,
            Some(st(&[("a", int, true, "{}")])),
        ),
        (
            st(&[("a", int, true, r#"{"comment":"x"}"#)]),
            st(&[("a", int, true, r#"{"comment":"x"}"#)]),
            true,
            Some(st(&[("a", int, true, r#"{"comment":"x"}"#)])),
        ),
        (
            st(&[("a", int, true, r#"{"comment":"x"}"#)]),
            st(&[("a", long, true, "{}")]),
            true,
            Some(st(&[("a", long, true, "{}")])),
        ),
    ];
    for (a, b, ansi, want) in cases {
        let want = want.as_ref().map(|w| Some(t(w).json()));
        assert_eq!(wider(a, b, *ansi), want, "{a} | {b} (ansi {ansi})");
    }
    // Comparisons meet by the tightest common type only.
    let compare = |a: &str, b: &str| {
        let (a, b) = (t(a), t(b));
        common(&arg(Some(&a)), &arg(Some(&b)), false, false).map(|r| r.map(|t| t.json()))
    };
    assert_eq!(
        compare(&arr(int, false), &arr(long, false)),
        Some(Some(arr(long, false)))
    );
    assert_eq!(compare(&arr(int, false), &arr(string, false)), None);
    assert_eq!(
        compare(&arr(int, false), &arr(r#""decimal(2,1)""#, false)),
        None
    );
    assert_eq!(
        compare(&arr(int, false), &arr(r#""decimal(20,0)""#, false)),
        Some(Some(arr(r#""decimal(20,0)""#, false)))
    );
}

/// Casts with nested types as Spark 4.2 resolves them on the oracle
/// (`SELECT CAST(...)`, ANSI on and off).
#[test]
fn nested_casts_follow_spark() {
    let t = |s: &str| krishiv_types::parse_data_type(s).unwrap();
    let not_null_a = |ty: T| -> T { StructType::new([StructField::new("a", ty, false)]).into() };
    type Outcome = Result<(), BadCast>;
    let cases: Vec<(T, T, Outcome, Outcome)> = vec![
        // (from, to, ANSI, legacy)
        (t("array<string>"), t("array<int>"), Ok(()), Ok(())),
        (
            T::String,
            t("array<int>"),
            Err(BadCast::Without),
            Err(BadCast::Without),
        ),
        (t("array<int>"), T::String, Ok(()), Ok(())),
        (
            t("map<string,int>"),
            t("map<int,int>"),
            Ok(()),
            Err(BadCast::Without),
        ),
        (t("map<int,string>"), t("map<bigint,int>"), Ok(()), Ok(())),
        (t("struct<a:int>"), t("struct<b:bigint>"), Ok(()), Ok(())),
        (
            t("struct<a:int>"),
            t("struct<a:int,b:int>"),
            Err(BadCast::Without),
            Err(BadCast::Without),
        ),
        (
            t("struct<a:int>"),
            not_null_a(T::Integer),
            Err(BadCast::Without),
            Err(BadCast::Without),
        ),
        (
            not_null_a(T::Integer),
            not_null_a(T::Integer),
            Ok(()),
            Ok(()),
        ),
        (
            not_null_a(T::String),
            not_null_a(T::Integer),
            Ok(()),
            Err(BadCast::Without),
        ),
        (
            t("array<int>"),
            t("array<date>"),
            Err(BadCast::Without),
            Err(BadCast::Without),
        ),
        (
            t("array<boolean>"),
            t("array<timestamp>"),
            Err(BadCast::Func("`try_cast`")),
            Ok(()),
        ),
        (
            t("array<int>"),
            t("array<binary>"),
            Err(BadCast::Func("`try_cast`")),
            Ok(()),
        ),
        (
            t("array<date>"),
            t("array<int>"),
            Err(BadCast::Func("`try_cast`")),
            Ok(()),
        ),
        (
            t("array<int>"),
            T::Integer,
            Err(BadCast::Without),
            Err(BadCast::Without),
        ),
        (
            t("array<int>"),
            t("map<int,int>"),
            Err(BadCast::Without),
            Err(BadCast::Without),
        ),
        (t("array<void>"), t("array<date>"), Ok(()), Ok(())),
        (t("array<void>"), t("array<array<int>>"), Ok(()), Ok(())),
    ];
    for (from, to, ansi, legacy) in cases {
        assert_eq!(
            can_cast(Some(&from), &to, CastMode::Ansi),
            ansi,
            "{from} -> {to} (ANSI)"
        );
        assert_eq!(
            can_cast(Some(&from), &to, CastMode::Legacy),
            legacy,
            "{from} -> {to}"
        );
    }
    assert_eq!(
        can_cast(
            Some(&t("array<boolean>")),
            &t("array<timestamp>"),
            CastMode::Try
        ),
        Ok(())
    );
}
