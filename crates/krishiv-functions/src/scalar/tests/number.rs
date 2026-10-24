// SPDX-License-Identifier: Apache-2.0
//! The number bindings: `hex`/`unhex`, `bin`, `conv`, `format_number`,
//! `to_number`/`try_to_number` and `to_char`/`to_varchar` over nulls,
//! edges and both ANSI modes.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Decimal128Type;
use arrow_array::{Array, ArrayRef, Decimal128Array, Int64Array, StringArray};
use arrow_schema::DataType;
use krishiv_common::error::ErrorClass;

use super::{
    BindArg, BindOptions, ScalarFunction, bind, bind_scalar, bind_with, bytes, class, col, ints,
    lit, run, strs, text,
};

fn longs(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn decimals(v: &[Option<i128>], p: u8, s: i8) -> ArrayRef {
    Arc::new(
        Decimal128Array::from(v.to_vec())
            .with_precision_and_scale(p, s)
            .unwrap(),
    )
}

fn legacy() -> BindOptions {
    BindOptions::new().with_ansi(false)
}

/// `name` over a string column and a string literal format: the bound
/// function and the format as an invocation argument.
fn with_format(
    name: &str,
    values: &ArrayRef,
    format: Option<&str>,
) -> (Arc<dyn ScalarFunction>, ArrayRef) {
    let fmt: ArrayRef = Arc::new(StringArray::from(vec![format]));
    let f = bind_scalar(
        name,
        &[
            values.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&fmt)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    (f, fmt)
}

fn bin_values(a: &ArrayRef) -> Vec<Option<Vec<u8>>> {
    a.as_binary::<i32>()
        .iter()
        .map(|b| b.map(<[u8]>::to_vec))
        .collect()
}

#[test]
fn hex_of_longs_binaries_and_strings() {
    // `SELECT hex(17)` is `11`; `SELECT hex('Spark SQL')` is its bytes.
    assert_eq!(
        text(&run("hex", &[&longs(&[Some(17), Some(0), Some(-1), None])])),
        super::owned(&[Some("11"), Some("0"), Some("FFFFFFFFFFFFFFFF"), None])
    );
    let b = bytes(&[Some(b"ab".as_slice()), None]);
    assert_eq!(
        text(&run("hex", &[&b])),
        super::owned(&[Some("6162"), None])
    );
    assert_eq!(
        text(&run("hex", &[&strs(&[Some("Spark SQL")])])),
        super::owned(&[Some("537061726B2053514C")])
    );
    // Ints take the long path too (through the analyzer's cast).
    assert_eq!(
        text(&run("hex", &[&longs(&[Some(28)])])),
        super::owned(&[Some("1C")])
    );
}

#[test]
fn unhex_pads_odd_lengths_and_nulls_the_rest() {
    // `hex(lpad(unhex('123'), 2))` is `0123`: odd lengths pad a nibble.
    let f = bind("unhex", &[&strs(&[Some("123")])]);
    assert_eq!(f.return_type(), &DataType::Binary);
    assert_eq!(
        bin_values(&run(
            "unhex",
            &[&strs(&[
                Some(""),
                Some("aabb"),
                Some("123"),
                Some("12345"),
                None
            ])]
        )),
        [
            Some(vec![]),
            Some(vec![0xaa, 0xbb]),
            Some(vec![0x01, 0x23]),
            Some(vec![0x01, 0x23, 0x45]),
            None,
        ]
    );
    // Anything else is NULL, not an error.
    assert_eq!(
        bin_values(&run(
            "unhex",
            &[&strs(&[Some("GG"), Some("12 34"), Some("0x12")])]
        )),
        [None, None, None]
    );
}

#[test]
fn bin_shows_longs_in_binary() {
    // The `math.sql` goldens.
    for (v, want) in [(0, "0"), (25, "11001")] {
        assert_eq!(
            text(&run("bin", &[&longs(&[Some(v)])])),
            super::owned(&[Some(want)])
        );
    }
    assert_eq!(
        text(&run("bin", &[&longs(&[Some(-13), None])])),
        super::owned(&[
            Some("1111111111111111111111111111111111111111111111111111111111110011"),
            None,
        ])
    );
}

#[test]
fn conv_goldens_and_bases() {
    // The `math.sql` goldens.
    let conv = |num: &str, from: i32, to: i32| {
        let (n, f, t) = (strs(&[Some(num)]), ints(&[Some(from)]), ints(&[Some(to)]));
        text(&run("conv", &[&n, &f, &t]))[0].clone()
    };
    assert_eq!(conv("100", 2, 10), Some("4".to_owned()));
    assert_eq!(conv("-10", 16, -10), Some("-16".to_owned()));
    assert_eq!(
        conv("9223372036854775808", 10, 16),
        Some("8000000000000000".to_owned())
    );
    // Overflow raises in ANSI mode...
    let (n, f, t) = (
        strs(&[Some("92233720368547758070")]),
        ints(&[Some(10)]),
        ints(&[Some(16)]),
    );
    let err = bind("conv", &[&n, &f, &t])
        .invoke(&[col(&n), col(&f), col(&t)], 1)
        .unwrap_err();
    assert_eq!(class(&err), ErrorClass::ArithmeticOverflow);
    assert_eq!(
        err.to_string(),
        "[ARITHMETIC_OVERFLOW] Overflow in function conv(). If necessary set \
         \"spark.sql.ansi.enabled\" to \"false\" to bypass this error. SQLSTATE: 22003"
    );
    // ... and formats `2^64 - 1` otherwise.
    let g = bind_with("conv", &[&n, &f, &t], &legacy());
    assert_eq!(
        text(&g.invoke(&[col(&n), col(&f), col(&t)], 1).unwrap()),
        super::owned(&[Some("FFFFFFFFFFFFFFFF")])
    );
    // Bad bases and empty input are NULL.
    for (from, to) in [(1, 10), (37, 10), (10, 1), (10, -37)] {
        assert_eq!(conv("100", from, to), None);
    }
    assert_eq!(conv("", 10, 16), None);
    assert_eq!(conv("  ", 10, 16), None);
    // Digits stop at the first invalid one; letters ignore case.
    assert_eq!(conv("12Z34", 10, 10), Some("12".to_owned()));
    assert_eq!(conv("ff", 16, 10), Some("255".to_owned()));
    assert_eq!(conv("FF", 16, 10), Some("255".to_owned()));
    // NULLs propagate.
    let (a, b) = (strs(&[None]), ints(&[Some(10)]));
    assert_eq!(text(&run("conv", &[&a, &b, &b])), super::owned(&[None]));
}

#[test]
fn format_number_int_mode() {
    // `SELECT format_number(12332.123456, 4)` is `12,332.1235`.
    let x = decimals(
        &[
            Some(12332123456),
            Some(12332123456),
            Some(12332123456),
            Some(12332123456),
            Some(12332123456),
        ],
        11,
        6,
    );
    let d = ints(&[Some(4), Some(1), Some(0), Some(-1), None]);
    let f = bind("format_number", &[&x, &d]);
    assert_eq!(f.return_type(), &DataType::Utf8);
    assert_eq!(
        text(&f.invoke(&[col(&x), col(&d)], 5).unwrap()),
        super::owned(&[
            Some("12,332.1235"),
            Some("12,332.1"),
            Some("12,332"),
            None,
            None,
        ])
    );
    // Longs group too; negative `d` is always NULL.
    assert_eq!(
        text(&run(
            "format_number",
            &[&longs(&[Some(1_234_567)]), &ints(&[Some(2)])]
        )),
        super::owned(&[Some("1,234,567.00")])
    );
}

#[test]
fn format_number_pattern_mode() {
    // `select format_number(ascii, '###.###')` in `collations-basic.sql`.
    let x = doubles(&[Some(66.421)]);
    let (f, fmt) = with_format("format_number", &x, Some("###.###"));
    assert_eq!(
        text(&f.invoke(&[col(&x), lit(&fmt)], 1).unwrap()),
        super::owned(&[Some("66.421")])
    );
    let x = decimals(&[Some(12332123)], 8, 3);
    let (f, fmt) = with_format("format_number", &x, Some("##################.###"));
    assert_eq!(
        text(&f.invoke(&[col(&x), lit(&fmt)], 1).unwrap()),
        super::owned(&[Some("12332.123")])
    );
    // `abc` is a bare prefix.
    let (f, fmt) = with_format("format_number", &x, Some("abc"));
    assert_eq!(
        text(&f.invoke(&[col(&x), lit(&fmt)], 1).unwrap()),
        super::owned(&[Some("abc12332")])
    );
    // Empty is the default pattern.
    let (f, fmt) = with_format("format_number", &x, Some(""));
    assert_eq!(
        text(&f.invoke(&[col(&x), lit(&fmt)], 1).unwrap()),
        super::owned(&[Some("12,332")])
    );
    // Outside the subset refuses loudly.
    let (f, fmt) = with_format("format_number", &x, Some("0.0E0"));
    let err = f.invoke(&[col(&x), lit(&fmt)], 1).unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
}

fn doubles(v: &[Option<f64>]) -> ArrayRef {
    Arc::new(arrow_array::Float64Array::from(v.to_vec()))
}

fn dec_values(a: &ArrayRef) -> Vec<Option<i128>> {
    a.as_primitive::<Decimal128Type>().iter().collect()
}

#[test]
fn to_number_goldens() {
    // `string-functions.sql`: every `to_number` statement.
    for (input, format, unscaled) in [
        ("454", "000", 454),
        ("454.2", "000.0", 4542),
        ("12,454", "00,000", 12454),
        ("$78.12", "$00.00", 7812),
        ("+454", "S000", 454),
        ("-454", "S000", -454),
        ("12,454.8-", "00,000.9MI", -124548),
        ("00,454.8-", "00,000.9MI", -4548),
        ("<00,454.8>", "00,000.9PR", -4548),
    ] {
        let v = strs(&[Some(input)]);
        let (f, fmt) = with_format("to_number", &v, Some(format));
        let out = f.invoke(&[col(&v), lit(&fmt)], 1).unwrap();
        assert_eq!(dec_values(&out), [Some(unscaled)], "{input} {format}");
    }
    // The decimal types follow the patterns.
    for (format, precision, scale) in [
        ("000", 3, 0),
        ("000.0", 4, 1),
        ("00,000", 5, 0),
        ("$00.00", 4, 2),
        ("00,000.9MI", 6, 1),
    ] {
        let v = strs(&[Some("0")]);
        let (f, _) = with_format("to_number", &v, Some(format));
        assert_eq!(
            f.return_type(),
            &DataType::Decimal128(precision, scale),
            "{format}"
        );
    }
    // NULL input is NULL; a NULL format is `decimal(10,0)` NULLs.
    let v = strs(&[None]);
    let (f, fmt) = with_format("to_number", &v, Some("000"));
    assert_eq!(
        dec_values(&f.invoke(&[col(&v), lit(&fmt)], 1).unwrap()),
        [None]
    );
    let v = strs(&[Some("1")]);
    let (f, fmt) = with_format("to_number", &v, None);
    assert_eq!(f.return_type(), &DataType::Decimal128(10, 0));
    assert_eq!(
        dec_values(&f.invoke(&[col(&v), lit(&fmt)], 1).unwrap()),
        [None]
    );
}

#[test]
fn to_number_mismatch_raises_and_try_returns_null() {
    let v = strs(&[Some("abc"), Some("012"), None]);
    let (f, fmt) = with_format("to_number", &v, Some("000"));
    let err = f.invoke(&[col(&v), lit(&fmt)], 3).unwrap_err();
    assert_eq!(class(&err), ErrorClass::InvalidFormatMismatchInput);
    assert_eq!(
        err.to_string(),
        "[INVALID_FORMAT.MISMATCH_INPUT] The format is invalid: 000. The input \"STRING\" abc \
         does not match the format. SQLSTATE: 42601"
    );
    let (g, fmt) = with_format("try_to_number", &v, Some("000"));
    assert_eq!(g.return_type(), &DataType::Decimal128(3, 0));
    assert_eq!(
        dec_values(&g.invoke(&[col(&v), lit(&fmt)], 3).unwrap()),
        [None, Some(12), None]
    );
}

#[test]
fn to_char_goldens() {
    // `charvarchar.sql`: `to_varchar` is `to_char`.
    for (unscaled, scale, format, want) in [
        (7812, 2, "$99.99", "$78.12"),
        (11111, 2, "99.9", "##.#"),
        (124548, 1, "99,999.9S", "12,454.8+"),
        (-124548, 1, "99G999D9S", "12,454.8-"),
    ] {
        let v = decimals(&[Some(unscaled)], 6, scale);
        for name in ["to_char", "to_varchar"] {
            let (f, fmt) = with_format(name, &v, Some(format));
            assert_eq!(
                text(&f.invoke(&[col(&v), lit(&fmt)], 1).unwrap()),
                super::owned(&[Some(want)]),
                "{name} {format}"
            );
        }
    }
    // NULLs propagate.
    let v = decimals(&[None], 6, 1);
    let (f, fmt) = with_format("to_char", &v, Some("99,999.9S"));
    assert_eq!(
        text(&f.invoke(&[col(&v), lit(&fmt)], 1).unwrap()),
        super::owned(&[None])
    );
}

#[test]
fn format_number_covers_widths_and_rejections() {
    // Every numeric width formats (nulls stay null).
    let one = ints(&[Some(1), None]);
    let i8s = Arc::new(arrow_array::Int8Array::from(vec![Some(1_i8), None])) as ArrayRef;
    assert_eq!(
        text(&run("format_number", &[&i8s, &one])),
        super::owned(&[Some("1.0"), None])
    );
    let i16s = Arc::new(arrow_array::Int16Array::from(vec![Some(1_i16), None])) as ArrayRef;
    assert_eq!(
        text(&run("format_number", &[&i16s, &one])),
        super::owned(&[Some("1.0"), None])
    );
    let f32s = Arc::new(arrow_array::Float32Array::from(vec![Some(1.5_f32), None])) as ArrayRef;
    assert_eq!(
        text(&run("format_number", &[&f32s, &one])),
        super::owned(&[Some("1.5"), None])
    );
    let nulls = Arc::new(arrow_array::NullArray::new(2)) as ArrayRef;
    assert_eq!(
        text(&run("format_number", &[&nulls, &one])),
        super::owned(&[None, None])
    );
    // A pattern that is neither an int nor a string is refused at bind time.
    let b = Arc::new(arrow_array::BooleanArray::from(vec![Some(true)])) as ArrayRef;
    let err = bind_scalar(
        "format_number",
        &[one.data_type().clone().into(), b.data_type().clone().into()],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn bad_bindings_are_rejected() {
    // Types the analyzer never produces are refused at bind time.
    let (i, s) = (ints(&[Some(1)]), strs(&[Some("a")]));
    for (name, bound) in [
        ("hex", vec![i.data_type().clone().into()]),
        ("unhex", vec![i.data_type().clone().into()]),
        ("bin", vec![s.data_type().clone().into()]),
        ("conv", vec![s.data_type().clone().into()]),
        (
            "to_number",
            vec![i.data_type().clone().into(), i.data_type().clone().into()],
        ),
        (
            "to_char",
            vec![i.data_type().clone().into(), s.data_type().clone().into()],
        ),
    ] {
        let bound: Vec<BindArg> = bound;
        let err = bind_scalar(name, &bound, &BindOptions::new()).unwrap_err();
        assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature, "{name}");
    }
    // A column format never reaches the kernel (the analyzer wants a literal).
    let fmt: ArrayRef = Arc::new(arrow_array::StringArray::from(vec![
        Some("000"),
        Some("000"),
    ]));
    let err = bind_scalar(
        "to_number",
        &[s.data_type().clone().into(), fmt.data_type().clone().into()],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn to_char_formats_small_fractions() {
    // A value smaller than the scale renders with a leading zero.
    let v = decimals(&[Some(5)], 4, 2);
    let (f, fmt) = with_format("to_char", &v, Some("99.99"));
    assert_eq!(
        text(&f.invoke(&[col(&v), lit(&fmt)], 1).unwrap()),
        super::owned(&[Some(" 0.05")])
    );
}

#[test]
fn format_number_int_values_and_null_formats() {
    // An `int` value formats too.
    assert_eq!(
        text(&run(
            "format_number",
            &[&ints(&[Some(42), None]), &ints(&[Some(0), Some(0)])]
        )),
        super::owned(&[Some("42"), None])
    );
    // A NULL format is all NULL (of the analyzer's `decimal(10,0)`).
    let v = strs(&[Some("1")]);
    let (f, fmt) = with_format("to_number", &v, None);
    assert_eq!(f.return_type(), &DataType::Decimal128(10, 0));
    assert_eq!(
        dec_values(&f.invoke(&[col(&v), lit(&fmt)], 1).unwrap()),
        [None]
    );
    let (f, fmt) = with_format("to_char", &decimals(&[Some(1)], 4, 0), None);
    assert_eq!(
        text(
            &f.invoke(&[col(&decimals(&[Some(1)], 4, 0)), lit(&fmt)], 1)
                .unwrap()
        ),
        super::owned(&[None])
    );
}
