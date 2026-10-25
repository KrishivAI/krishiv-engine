// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Int32Type, TimestampMicrosecondType};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Date32Array, Int8Array, Int16Array, Int32Array,
    LargeStringArray, StringArray, StringViewArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_kernels::cast::FixedOffset;
use krishiv_kernels::{KernelCtx, SparkVersion};

use super::{Arg, BindArg, BindOptions, ScalarFunction, bind_scalar, scalar_names};

pub(super) fn strs(v: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(v.to_vec()))
}

pub(super) fn ints(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(v.to_vec()))
}

fn dates(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Date32Array::from(v.to_vec()))
}

pub(super) fn bytes(v: &[Option<&[u8]>]) -> ArrayRef {
    Arc::new(BinaryArray::from(v.to_vec()))
}

fn ltz(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(TimestampMicrosecondArray::from(v.to_vec()).with_timezone("UTC"))
}

fn ntz(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(TimestampMicrosecondArray::from(v.to_vec()))
}

pub(super) fn col(a: &ArrayRef) -> Arg {
    Arg::Array(Arc::clone(a))
}

pub(super) fn lit(a: &ArrayRef) -> Arg {
    Arg::Scalar(Arc::clone(a))
}

pub(super) fn types(args: &[&ArrayRef]) -> Vec<BindArg> {
    args.iter().map(|a| a.data_type().clone().into()).collect()
}

pub(super) fn bind_with(
    name: &str,
    args: &[&ArrayRef],
    opts: &BindOptions,
) -> Arc<dyn ScalarFunction> {
    bind_scalar(name, &types(args), opts).unwrap()
}

pub(super) fn bind(name: &str, args: &[&ArrayRef]) -> Arc<dyn ScalarFunction> {
    bind_with(name, args, &BindOptions::new())
}

/// `name` over columns `args`.
pub(super) fn run(name: &str, args: &[&ArrayRef]) -> ArrayRef {
    let rows = args.first().map_or(1, |a| Array::len(a.as_ref()));
    let f = bind(name, args);
    f.invoke(&args.iter().map(|a| col(a)).collect::<Vec<_>>(), rows)
        .unwrap()
}

pub(super) fn text(a: &ArrayRef) -> Vec<Option<String>> {
    a.as_string::<i32>()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect()
}

pub(super) fn owned(v: &[Option<&str>]) -> Vec<Option<String>> {
    v.iter().map(|s| s.map(str::to_owned)).collect()
}

pub(super) fn int_values(a: &ArrayRef) -> Vec<Option<i32>> {
    a.as_primitive::<Int32Type>().iter().collect()
}

fn bools(a: &ArrayRef) -> Vec<Option<bool>> {
    a.as_boolean().iter().collect()
}

pub(super) fn class(e: &KrishivError) -> ErrorClass {
    e.error_class()
}

#[test]
fn names_are_sorted_and_include_aliases() {
    let names = scalar_names();
    assert!(names.windows(2).all(|w| w[0] < w[1]));
    for n in ["upper", "ucase", "substr", "dateadd", "now", "date_part"] {
        assert!(names.contains(&n), "{n}");
    }
}

#[test]
fn case_and_lengths() {
    let a = strs(&[Some("abc"), None, Some("straße")]);
    let up = bind("UPPER", &[&a]);
    assert_eq!(up.name(), "upper");
    assert_eq!(up.return_type(), &DataType::Utf8);
    assert_eq!(
        text(&up.invoke(&[col(&a)], 3).unwrap()),
        owned(&[Some("ABC"), None, Some("STRASSE")])
    );
    assert_eq!(
        text(&run("lcase", &[&strs(&[Some("AbC")])])),
        owned(&[Some("abc")])
    );
    assert_eq!(int_values(&run("length", &[&a])), [Some(3), None, Some(6)]);
    let b = bytes(&[Some(b"\xff\x00"), None]);
    assert_eq!(int_values(&run("char_length", &[&b])), [Some(2), None]);
    // The Unicode data follows the Spark version (U+A7CE maps only in 4.2).
    let new_mapping = strs(&[Some("\u{A7CE}")]);
    let v41 =
        BindOptions::new().with_kernel(KernelCtx::new().with_spark_version(SparkVersion::V4_1));
    let f = bind_with("lower", &[&new_mapping], &v41);
    let g = bind("lower", &[&new_mapping]);
    let (old, new) = (
        f.invoke(&[col(&new_mapping)], 1).unwrap(),
        g.invoke(&[col(&new_mapping)], 1).unwrap(),
    );
    assert_ne!(text(&old), text(&new));
}

#[test]
fn storage_alternatives_are_read() {
    let view: ArrayRef = Arc::new(StringViewArray::from(vec![Some("ab"), None]));
    let large: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("ab"), None]));
    for a in [&view, &large] {
        assert_eq!(text(&run("upper", &[a])), owned(&[Some("AB"), None]));
    }
    let bview: ArrayRef = Arc::new(BinaryViewArray::from(vec![Some(b"ab".as_slice()), None]));
    assert_eq!(int_values(&run("length", &[&bview])), [Some(2), None]);
    let large_bin: ArrayRef = Arc::new(arrow_array::LargeBinaryArray::from(vec![Some(
        b"abc".as_slice(),
    )]));
    assert_eq!(int_values(&run("length", &[&large_bin])), [Some(3)]);
}

#[test]
fn trims_take_spark_argument_orders() {
    let s = strs(&[Some("xxaxx"), Some("  b  "), None]);
    let x = strs(&[Some("x")]);
    assert_eq!(
        text(&run("trim", &[&s])),
        owned(&[Some("xxaxx"), Some("b"), None])
    );
    // `trim(chars, str)`: the characters first.
    let f = bind("ltrim", &[&x, &s]);
    assert_eq!(
        text(&f.invoke(&[lit(&x), col(&s)], 3).unwrap()),
        owned(&[Some("axx"), Some("  b  "), None])
    );
    let f = bind("rtrim", &[&x, &s]);
    assert_eq!(
        text(&f.invoke(&[lit(&x), col(&s)], 3).unwrap()),
        owned(&[Some("xxa"), Some("  b  "), None])
    );
    // `btrim(str, chars)`: the characters last.
    let f = bind("btrim", &[&s, &x]);
    assert_eq!(
        text(&f.invoke(&[col(&s), lit(&x)], 3).unwrap()),
        owned(&[Some("a"), Some("  b  "), None])
    );
}

#[test]
fn matches_on_strings_and_bytes() {
    let s = strs(&[Some("spark"), Some("sql"), None]);
    let p = strs(&[Some("sp")]);
    let f = bind("startswith", &[&s, &p]);
    assert_eq!(f.return_type(), &DataType::Boolean);
    assert_eq!(
        bools(&f.invoke(&[col(&s), lit(&p)], 3).unwrap()),
        [Some(true), Some(false), None]
    );
    assert_eq!(
        bools(&run(
            "endswith",
            &[&s, &strs(&[Some("rk"), Some("rk"), Some("")])]
        )),
        [Some(true), Some(false), None]
    );
    let b = bytes(&[Some(b"\x01\x02\x03"), None]);
    let n = bytes(&[Some(b"\x02"), Some(b"\x02")]);
    assert_eq!(bools(&run("contains", &[&b, &n])), [Some(true), None]);
}

#[test]
fn replace_with_and_without_replacement() {
    let s = strs(&[Some("abcabc"), None]);
    let search = strs(&[Some("b"), Some("b")]);
    assert_eq!(
        text(&run("replace", &[&s, &search])),
        owned(&[Some("acac"), None])
    );
    let with = strs(&[Some("XY"), Some("XY")]);
    assert_eq!(
        text(&run("replace", &[&s, &search, &with])),
        owned(&[Some("aXYcaXYc"), None])
    );
}

#[test]
fn pads_on_strings_and_bytes() {
    let s = strs(&[Some("hi"), None]);
    let n = ints(&[Some(5), Some(5)]);
    assert_eq!(text(&run("lpad", &[&s, &n])), owned(&[Some("   hi"), None]));
    let fill = strs(&[Some("?!"), Some("?!")]);
    assert_eq!(
        text(&run("rpad", &[&s, &n, &fill])),
        owned(&[Some("hi?!?"), None])
    );
    let b = bytes(&[Some(b"\x01")]);
    let three = ints(&[Some(3)]);
    let out = run("lpad", &[&b, &three]);
    assert_eq!(out.data_type(), &DataType::Binary);
    assert_eq!(out.as_binary::<i32>().value(0), b"\x00\x00\x01");
    let f = bytes(&[Some(b"\xee")]);
    let out = run("rpad", &[&b, &three, &f]);
    assert_eq!(out.as_binary::<i32>().value(0), b"\x01\xee\xee");
}

#[test]
fn substrings_in_characters_and_bytes() {
    let s = strs(&[Some("Spark SQL"), Some("ßa"), None]);
    let pos = ints(&[Some(5)]);
    let f = bind("substr", &[&s, &pos]);
    assert_eq!(f.name(), "substr");
    assert_eq!(
        text(&f.invoke(&[col(&s), lit(&pos)], 3).unwrap()),
        owned(&[Some("k SQL"), Some(""), None])
    );
    let (one, len) = (ints(&[Some(1)]), ints(&[Some(1)]));
    let f = bind("substring", &[&s, &one, &len]);
    assert_eq!(
        text(&f.invoke(&[col(&s), lit(&one), lit(&len)], 3).unwrap()),
        owned(&[Some("S"), Some("ß"), None])
    );
    // A `smallint` or `tinyint` position is widened.
    let small: ArrayRef = Arc::new(Int16Array::from(vec![Some(-3)]));
    let f = bind("substring", &[&s, &small]);
    assert_eq!(
        text(&f.invoke(&[col(&s), lit(&small)], 3).unwrap()),
        owned(&[Some("SQL"), Some("ßa"), None])
    );
    let b = bytes(&[Some(b"\x01\x02\x03")]);
    let tiny: ArrayRef = Arc::new(Int8Array::from(vec![Some(2)]));
    let out = run("substring", &[&b, &tiny, &one]);
    assert_eq!(out.as_binary::<i32>().value(0), b"\x02");
    let out = run("substring", &[&b, &tiny]);
    assert_eq!(out.as_binary::<i32>().value(0), b"\x02\x03");
}

#[test]
fn literal_arguments_are_broadcast() {
    // Every argument a literal: one row is computed and repeated.
    let s = strs(&[Some("ab")]);
    let f = bind("upper", &[&s]);
    assert_eq!(
        text(&f.invoke(&[lit(&s)], 3).unwrap()),
        owned(&[Some("AB"), Some("AB"), Some("AB")])
    );
    assert_eq!(
        text(&f.invoke(&[lit(&s)], 1).unwrap()),
        owned(&[Some("AB")])
    );
    assert_eq!(f.invoke(&[lit(&s)], 0).unwrap().len(), 0);
    // An empty batch computes nothing, so a literal that would overflow
    // does not raise.
    let (max, one) = (dates(&[Some(i32::MAX)]), ints(&[Some(1)]));
    let add = bind("date_add", &[&max, &one]);
    let out = add.invoke(&[lit(&max), lit(&one)], 0).unwrap();
    assert_eq!((out.len(), out.data_type()), (0, &DataType::Date32));
    assert!(add.invoke(&[lit(&max), lit(&one)], 1).is_err());
    let null = strs(&[None]);
    let p = ints(&[Some(1)]);
    let f = bind("substring", &[&null, &p]);
    let out = f.invoke(&[lit(&null), lit(&p)], 2).unwrap();
    assert_eq!(out.null_count(), 2);
    // A literal string against a column of positions.
    let pos = ints(&[Some(1), Some(2), None]);
    let out = bind("substring", &[&s, &pos])
        .invoke(&[lit(&s), col(&pos)], 3)
        .unwrap();
    assert_eq!(text(&out), owned(&[Some("ab"), Some("b"), None]));
}

#[test]
fn bind_errors() {
    let opts = BindOptions::default();
    let err = bind_scalar("no_such_function", &[], &opts).unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("no_such_function"), "{err}");
    let err = bind_scalar("upper", &[DataType::Int32.into()], &opts).unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("upper(Int32)"), "{err}");
    let err = bind_scalar("date_add", &[DataType::Date32.into()], &opts).unwrap_err();
    assert!(err.to_string().contains("date_add(Date32)"), "{err}");
    let err = bind_scalar(
        "lpad",
        &[DataType::Float64.into(), DataType::Int32.into()],
        &opts,
    )
    .unwrap_err();
    assert!(err.to_string().contains("lpad(Float64, Int32)"), "{err}");
}

#[test]
fn invoke_checks_its_arguments() {
    let s = strs(&[Some("a"), Some("b")]);
    let f = bind("upper", &[&s]);
    let internal = |e: KrishivError| {
        assert_eq!(class(&e), ErrorClass::InternalError);
        e.to_string()
    };
    // Wrong argument count, rows, type.
    assert!(internal(f.invoke(&[], 2).unwrap_err()).contains("called with 0"));
    assert!(internal(f.invoke(&[col(&s)], 3).unwrap_err()).contains("2 rows"));
    assert!(internal(f.invoke(&[lit(&s)], 2).unwrap_err()).contains("where 1 rows"));
    let i = ints(&[Some(1), Some(2)]);
    assert!(internal(f.invoke(&[col(&i)], 2).unwrap_err()).contains("Int32"));
    let dbg = format!("{f:?}");
    assert!(dbg.contains("upper") && dbg.contains("Utf8"), "{dbg}");
    assert_eq!(Arg::Array(Arc::clone(&s)).array().len(), 2);
}

#[test]
fn bind_arg_types() {
    let l = BindArg::Literal(strs(&[Some("x")]));
    assert_eq!(l.data_type(), DataType::Utf8);
    assert_eq!(
        BindArg::from(DataType::Date32).data_type(),
        DataType::Date32
    );
}

#[test]
fn day_arithmetic_follows_ansi() {
    let d = dates(&[Some(0), Some(i32::MAX), None]);
    let n = ints(&[Some(1)]);
    let f = bind("date_add", &[&d, &n]);
    assert_eq!(f.return_type(), &DataType::Date32);
    let err = f.invoke(&[col(&d), lit(&n)], 3).unwrap_err();
    assert_eq!(class(&err), ErrorClass::ArithmeticOverflow);
    // ANSI off: Java `int` wrap.
    let legacy = BindOptions::new().with_ansi(false);
    let f = bind_with("dateadd", &[&d, &n], &legacy);
    assert_eq!(f.name(), "dateadd");
    let out = f.invoke(&[col(&d), lit(&n)], 3).unwrap();
    assert_eq!(
        out.as_primitive::<Date32Type>().iter().collect::<Vec<_>>(),
        [Some(1), Some(i32::MIN), None]
    );
    let small: ArrayRef = Arc::new(Int8Array::from(vec![Some(2)]));
    let out = run("date_sub", &[&dates(&[Some(10)]), &small]);
    assert_eq!(
        out.as_primitive::<Date32Type>().iter().collect::<Vec<_>>(),
        [Some(8)]
    );
    let out = run(
        "datediff",
        &[&dates(&[Some(10), None]), &dates(&[Some(3), Some(1)])],
    );
    assert_eq!(int_values(&out), [Some(7), None]);
    let out = run(
        "date_diff",
        &[&dates(&[Some(-1)]), &dates(&[Some(i32::MAX)])],
    );
    assert_eq!(int_values(&out), [Some(i32::MIN)]);
}

#[test]
fn fields_of_dates_and_timestamps() {
    // 2024-03-05 is day 19787.
    let d = dates(&[Some(19_787), None]);
    assert_eq!(int_values(&run("year", &[&d])), [Some(2024), None]);
    assert_eq!(int_values(&run("month", &[&d])), [Some(3), None]);
    assert_eq!(int_values(&run("dayofmonth", &[&d])), [Some(5), None]);
    // 2024-03-05T23:30:00Z; in UTC+01:00 it is already the 6th.
    let micros = 1_709_681_400_000_000_i64;
    let ts = ltz(&[Some(micros)]);
    assert_eq!(int_values(&run("day", &[&ts])), [Some(5)]);
    let plus_one = BindOptions::new().with_time_zone(Arc::new(FixedOffset(3600)));
    let f = bind_with("day", &[&ts], &plus_one);
    assert_eq!(int_values(&f.invoke(&[col(&ts)], 1).unwrap()), [Some(6)]);
    // A timestamp_ntz is a local date-time: no zone applies.
    let local = ntz(&[Some(micros)]);
    let f = bind_with("day", &[&local], &plus_one);
    assert_eq!(int_values(&f.invoke(&[col(&local)], 1).unwrap()), [Some(5)]);
}

#[test]
fn extract_takes_a_literal_field() {
    let ts = ltz(&[Some(1_709_681_400_123_456), None]);
    let field = strs(&[Some("second")]);
    let args = [
        BindArg::Literal(Arc::clone(&field)),
        ts.data_type().clone().into(),
    ];
    let f = bind_scalar("date_part", &args, &BindOptions::new()).unwrap();
    assert_eq!(f.return_type(), &DataType::Decimal128(8, 6));
    let out = f.invoke(&[lit(&field), col(&ts)], 2).unwrap();
    let out = out.as_primitive::<arrow_array::types::Decimal128Type>();
    assert_eq!(out.iter().collect::<Vec<_>>(), [Some(123_456), None]);
    let year = strs(&[Some("YEAR")]);
    let args = [BindArg::Literal(Arc::clone(&year)), DataType::Date32.into()];
    let f = bind_scalar("extract", &args, &BindOptions::new()).unwrap();
    let d = dates(&[Some(19_787)]);
    assert_eq!(
        int_values(&f.invoke(&[lit(&year), col(&d)], 1).unwrap()),
        [Some(2024)]
    );
    let view: ArrayRef = Arc::new(StringViewArray::from(vec!["month"]));
    let args = [BindArg::Literal(view), DataType::Date32.into()];
    assert!(bind_scalar("extract", &args, &BindOptions::new()).is_ok());
    let large: ArrayRef = Arc::new(LargeStringArray::from(vec!["dow"]));
    let args = [BindArg::Literal(large), DataType::Date32.into()];
    assert!(bind_scalar("extract", &args, &BindOptions::new()).is_ok());
}

#[test]
fn a_null_extract_field_is_a_null_double() {
    let null = strs(&[None]);
    let args = [BindArg::Literal(Arc::clone(&null)), DataType::Utf8.into()];
    let f = bind_scalar("date_part", &args, &BindOptions::new()).unwrap();
    assert_eq!(f.return_type(), &DataType::Float64);
    let s = strs(&[Some("x"), None]);
    let out = f.invoke(&[lit(&null), col(&s)], 2).unwrap();
    assert_eq!((out.len(), out.null_count()), (2, 2));
}

#[test]
fn extract_rejects_unknown_or_unbound_fields() {
    let bad = |field: BindArg| {
        let args = [field, DataType::Date32.into()];
        let err = bind_scalar("extract", &args, &BindOptions::new()).unwrap_err();
        assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
        err.to_string()
    };
    assert!(bad(BindArg::Literal(strs(&[Some("fortnight")]))).contains("fortnight"));
    let args = [
        BindArg::Literal(strs(&[Some("year")])),
        DataType::Int32.into(),
    ];
    let err = bind_scalar("extract", &args, &BindOptions::new()).unwrap_err();
    assert!(err.to_string().contains("extract(Utf8, Int32)"), "{err}");
    assert!(bad(DataType::Utf8.into()).contains("not a literal"));
    // A literal is one row.
    for rows in [0, 2] {
        let field = BindArg::Literal(strs(&vec![Some("year"); rows]));
        let err = bind_scalar(
            "extract",
            &[field, DataType::Date32.into()],
            &BindOptions::new(),
        )
        .unwrap_err();
        assert_eq!(class(&err), ErrorClass::InternalError);
        assert!(err.to_string().contains(&format!("{rows} rows")), "{err}");
    }
}

#[test]
fn the_query_clock() {
    // 2024-03-05T23:30:00Z.
    let micros = 1_709_681_400_000_000_i64;
    let opts = BindOptions::new()
        .with_query_start(micros)
        .with_time_zone(Arc::new(FixedOffset(3600)));
    let f = bind_scalar("curdate", &[], &opts).unwrap();
    let out = f.invoke(&[], 2).unwrap();
    assert_eq!(
        out.as_primitive::<Date32Type>().iter().collect::<Vec<_>>(),
        [Some(19_788), Some(19_788)]
    );
    let f = bind_scalar("now", &[], &opts).unwrap();
    assert_eq!(
        f.return_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC")))
    );
    let out = f.invoke(&[], 1).unwrap();
    assert_eq!(
        out.as_primitive::<TimestampMicrosecondType>().value(0),
        micros
    );
    // A start instant outside the date range fails at bind.
    let far = opts.with_query_start(i64::MAX);
    assert!(bind_scalar("current_date", &[], &far).is_err());
}

#[test]
fn binary_literals_and_mixed_shapes() {
    let b = bytes(&[Some(b"\x01\x02"), Some(b"\x03"), None]);
    let needle = bytes(&[Some(b"\x02")]);
    let f = bind("endswith", &[&b, &needle]);
    assert_eq!(
        bools(&f.invoke(&[col(&b), lit(&needle)], 3).unwrap()),
        [Some(true), Some(false), None]
    );
    let null = bytes(&[None]);
    assert_eq!(
        bools(&f.invoke(&[col(&b), lit(&null)], 3).unwrap()),
        [None, None, None]
    );
    // A literal date with a column of day counts.
    let d = dates(&[Some(5)]);
    let n = ints(&[Some(1), None]);
    let f = bind("date_add", &[&d, &n]);
    let out = f.invoke(&[lit(&d), col(&n)], 2).unwrap();
    assert_eq!(
        out.as_primitive::<Date32Type>().iter().collect::<Vec<_>>(),
        [Some(6), None]
    );
}

#[test]
fn argument_conversions_reject_other_types() {
    use super::args::Inputs;
    let i = ints(&[Some(1)]);
    let s = strs(&[Some("a")]);
    let args = [col(&i), lit(&s)];
    let x = Inputs::new(&args, 1);
    let internal = |e: KrishivError| {
        assert_eq!(class(&e), ErrorClass::InternalError);
        e.to_string()
    };
    assert!(internal(x.str(0).err().unwrap()).contains("string argument of type Int32"));
    assert!(internal(x.bin(0).err().unwrap()).contains("binary argument of type Int32"));
    assert!(internal(x.date(1).err().unwrap()).contains("read as Date32"));
    assert!(internal(x.int(1).err().unwrap()).contains("int argument of type Utf8"));
    assert!(internal(x.dates(0).err().unwrap()).contains("date argument"));
    assert!(internal(x.timestamps(0).err().unwrap()).contains("timestamp argument"));
    assert!(internal(x.column(2).err().unwrap()).contains("missing argument 2"));
    // A literal read as a column is repeated.
    assert_eq!(x.strings(1).unwrap().len(), 1);
    let wide = Inputs::new(&args, 4);
    assert_eq!(wide.column(1).unwrap().len(), 4);
}

#[test]
fn results_are_checked_against_the_binding() {
    let wrong = super::Bound {
        name: "wrong",
        types: vec![],
        return_type: DataType::Int32,
        body: Box::new(|_| Ok(Arc::new(StringArray::from(vec!["x"])) as ArrayRef)),
    };
    let err = wrong.invoke(&[], 1).unwrap_err();
    assert_eq!(class(&err), ErrorClass::InternalError);
    assert!(err.to_string().contains("returned 1 rows of Utf8"), "{err}");
}

mod bitwise;
mod calendar;
mod families;
mod json;
mod number;
mod predicate;

#[test]
fn session_user_and_local_clock() {
    use arrow_array::cast::AsArray;
    use arrow_array::types::TimestampMicrosecondType;
    let opts = BindOptions::new().with_user("etl");
    for name in ["current_user", "USER", "session_user"] {
        let f = bind_scalar(name, &[], &opts).unwrap();
        assert_eq!(f.return_type(), &DataType::Utf8);
        let out = f.invoke(&[], 3).unwrap();
        let out: Vec<_> = out.as_string::<i32>().iter().collect();
        assert_eq!(out, [Some("etl"), Some("etl"), Some("etl")]);
    }
    assert!(bind_scalar("current_user", &[BindArg::Column(DataType::Int32)], &opts).is_err());
    // `localtimestamp` is the wall clock of the session zone, without a zone.
    let micros = 1_700_000_000_123_456;
    let opts = BindOptions::new()
        .with_query_start(micros)
        .with_time_zone(Arc::new(FixedOffset(19_800)));
    let f = bind_scalar("localtimestamp", &[], &opts).unwrap();
    assert_eq!(
        f.return_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    let out = f.invoke(&[], 2).unwrap();
    assert_eq!(
        out.as_primitive::<TimestampMicrosecondType>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(micros + 19_800_000_000); 2]
    );
}
