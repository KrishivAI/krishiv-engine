// SPDX-License-Identifier: Apache-2.0
//! The signatures of the character-level and encoding string functions:
//! which Spark release lines have them, how their arguments are cast and
//! when their result may be NULL.

use krishiv_common::error::ErrorClass;
use krishiv_types::DataType;

use super::{def, null_ty, result, ty};
use crate::registry::{Execution, Lookup, builtins};
use crate::signature::{Constant, FunctionKind, SparkProfile};

const SINCE_3_5: &[&str] = &[
    "ascii",
    "base64",
    "bit_length",
    "char",
    "chr",
    "concat_ws",
    "decode",
    "elt",
    "encode",
    "find_in_set",
    "initcap",
    "instr",
    "left",
    "len",
    "levenshtein",
    "locate",
    "luhn_check",
    "octet_length",
    "overlay",
    "position",
    "repeat",
    "reverse",
    "right",
    "soundex",
    "space",
    "split_part",
    "substring_index",
    "to_binary",
    "translate",
    "try_to_binary",
    "unbase64",
    "url_decode",
    "url_encode",
];

const SINCE_4_0: &[&str] = &[
    "is_valid_utf8",
    "make_valid_utf8",
    "validate_utf8",
    "try_validate_utf8",
    "try_url_decode",
];

#[test]
fn availability_follows_the_release_lines() {
    let r = builtins();
    for name in SINCE_3_5 {
        for p in SparkProfile::ALL {
            assert!(
                matches!(r.lookup(name, p), Lookup::Found(_)),
                "{name} {p:?}"
            );
        }
    }
    for name in SINCE_4_0 {
        assert!(
            matches!(r.lookup(name, SparkProfile::V3_5), Lookup::NotFound),
            "{name}"
        );
        for p in [SparkProfile::V4_0, SparkProfile::V4_1, SparkProfile::V4_2] {
            assert!(
                matches!(r.lookup(name, p), Lookup::Found(_)),
                "{name} {p:?}"
            );
        }
    }
    // `quote` arrived in 4.1.
    for p in [SparkProfile::V3_5, SparkProfile::V4_0] {
        assert!(matches!(r.lookup("quote", p), Lookup::NotFound));
    }
    for p in [SparkProfile::V4_1, SparkProfile::V4_2] {
        assert!(matches!(r.lookup("QUOTE", p), Lookup::Found(d) if d.name == "quote"));
    }
    // What stays unimplemented is known, not missing.
    for name in [
        "format_string",
        "printf",
        "parse_url",
        "sentences",
        "split",
        "collate",
    ] {
        assert!(
            matches!(r.lookup(name, SparkProfile::V4_2), Lookup::KnownUnsupported),
            "{name}"
        );
    }
}

#[test]
fn every_function_is_a_deterministic_scalar() {
    for name in SINCE_3_5.iter().chain(SINCE_4_0).chain(&["quote"]) {
        let d = def(name);
        assert_eq!(
            (d.kind, d.execution),
            (FunctionKind::Scalar, Execution::Scalar),
            "{name}"
        );
        assert!(d.deterministic, "{name}");
    }
}

#[test]
fn results_are_nullable_as_spark_has_them() {
    let s = || ty(DataType::String);
    let n = || ty(DataType::Integer);
    let t = |dt: DataType, nullable: bool| crate::signature::ArgType::new(dt, nullable);
    // Typed by the arguments.
    assert_eq!(
        result("ascii", &[s()], &[]).unwrap(),
        t(DataType::Integer, false)
    );
    assert_eq!(
        result("ascii", &[null_ty()], &[]).unwrap(),
        t(DataType::Integer, true)
    );
    assert_eq!(
        result("left", &[ty(DataType::Binary), n()], &[]).unwrap(),
        t(DataType::Binary, false)
    );
    // Always nullable.
    for (name, args, dt) in [
        ("right", vec![s(), n()], DataType::String),
        ("split_part", vec![s(), s(), n()], DataType::String),
        ("quote", vec![s()], DataType::String),
        ("luhn_check", vec![s()], DataType::Boolean),
        ("encode", vec![s(), s()], DataType::Binary),
        ("elt", vec![n(), s(), s()], DataType::String),
        ("url_encode", vec![s()], DataType::String),
        ("try_validate_utf8", vec![s()], DataType::String),
    ] {
        assert_eq!(result(name, &args, &[]).unwrap(), t(dt, true), "{name}");
    }
    // `locate` is NULL for a NULL substring or string, not a NULL start;
    // `concat_ws` only for a NULL separator.
    let nullable = |dt| t(dt, true);
    let a = [s(), s(), nullable(DataType::Integer)];
    assert_eq!(
        result("locate", &a, &[]).unwrap(),
        t(DataType::Integer, false)
    );
    let a = [nullable(DataType::String), s(), n()];
    assert_eq!(
        result("position", &a, &[]).unwrap(),
        t(DataType::Integer, true)
    );
    let a = [s(), nullable(DataType::String)];
    assert_eq!(
        result("concat_ws", &a, &[]).unwrap(),
        t(DataType::String, false)
    );
    let a = [nullable(DataType::String), s()];
    assert_eq!(
        result("concat_ws", &a, &[]).unwrap(),
        t(DataType::String, true)
    );
    // `base64` never gives NULL for a value.
    assert_eq!(
        result("base64", &[ty(DataType::Binary)], &[]).unwrap(),
        t(DataType::String, false)
    );
}

#[test]
fn to_binary_formats_are_checked_while_typing() {
    let args = [ty(DataType::String), ty(DataType::String)];
    let fmt = |name: &str, c: Option<Constant>| result(name, &args, &[None, c]);
    for f in ["hex", "UTF-8", "utf8", "Base64"] {
        assert!(
            fmt("to_binary", Some(Constant::String(f.to_owned()))).is_ok(),
            "{f}"
        );
    }
    assert!(fmt("to_binary", Some(Constant::Null)).is_ok());
    let err = fmt("to_binary", Some(Constant::String("Bogus".to_owned()))).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchInvalidArgValue
    );
    let shown: Vec<_> = err
        .message_parameters()
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    assert!(
        shown.contains(&"inputValue='bogus'".to_owned()),
        "{shown:?}"
    );
    let err = fmt("to_binary", Some(Constant::Integral(1))).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::DatatypeMismatchInvalidArgValue
    );
    // A format that is not a constant is refused by both.
    for name in ["to_binary", "try_to_binary"] {
        let err = fmt(name, None).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchNonFoldableInput,
            "{name}"
        );
    }
    // `try_to_binary` turns an unknown format into NULL instead.
    assert!(fmt("try_to_binary", Some(Constant::String("Bogus".to_owned()))).is_ok());
    assert!(result("to_binary", &args[..1], &[]).is_ok());
}

#[test]
fn decode_takes_two_arguments() {
    let two = [ty(DataType::Binary), ty(DataType::String)];
    assert_eq!(
        result("decode", &two, &[]).unwrap().data_type,
        Some(DataType::String)
    );
    let many = [
        ty(DataType::Integer),
        ty(DataType::Integer),
        ty(DataType::String),
    ];
    let err = result("decode", &many, &[]).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn argument_casts_follow_spark() {
    use crate::signature::{CallTypes, InputType};
    let inputs = |name: &str, args: &[crate::signature::ArgType]| {
        def(name)
            .inputs
            .expected(&CallTypes::new(args, &[]))
            .unwrap()
    };
    let (s, b, i) = (
        ty(DataType::String),
        ty(DataType::Binary),
        ty(DataType::Integer),
    );
    // `overlay` and `elt` choose binary when every text argument is binary.
    assert_eq!(
        inputs("overlay", &[b.clone(), b.clone(), i.clone()])[0],
        InputType::Exact(DataType::Binary)
    );
    assert_eq!(
        inputs("overlay", &[b.clone(), s.clone(), i.clone()])[0],
        InputType::Exact(DataType::String)
    );
    assert_eq!(
        inputs("elt", &[i.clone(), b.clone()])[1],
        InputType::Exact(DataType::Binary)
    );
    assert_eq!(
        inputs("elt", &[i, b, s.clone()])[1],
        InputType::Exact(DataType::String)
    );
    assert_eq!(
        inputs("decode", &[s.clone(), s.clone()])[0],
        InputType::Exact(DataType::Binary)
    );
    assert_eq!(
        inputs("decode", &[s.clone(), s.clone(), s])[0],
        InputType::Any
    );
}
