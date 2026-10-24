// SPDX-License-Identifier: Apache-2.0
//! The bitwise bindings: `&` `|` `^` `~`, the shifts, `bit_count` and
//! `bit_get`/`getbit` over nulls, edges and both ANSI modes.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int8Array, Int16Array, Int64Array};
use arrow_schema::DataType;
use krishiv_common::error::ErrorClass;

use super::{BindArg, BindOptions, bind, bind_scalar, bind_with, class, col, ints, run};

fn i8s(v: &[Option<i8>]) -> ArrayRef {
    Arc::new(Int8Array::from(v.to_vec()))
}

fn i16s(v: &[Option<i16>]) -> ArrayRef {
    Arc::new(Int16Array::from(v.to_vec()))
}

fn longs(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn i32s(a: &ArrayRef) -> Vec<Option<i32>> {
    a.as_primitive::<Int32Type>().iter().collect()
}

fn i8vals(a: &ArrayRef) -> Vec<Option<i8>> {
    a.as_primitive::<Int8Type>().iter().collect()
}

fn i64vals(a: &ArrayRef) -> Vec<Option<i64>> {
    a.as_primitive::<Int64Type>().iter().collect()
}

fn legacy() -> BindOptions {
    BindOptions::new().with_ansi(false)
}

#[test]
fn bitwise_and_or_xor_nulls_propagate() {
    let l = ints(&[Some(3), Some(0), None, Some(-1)]);
    let r = ints(&[Some(5), None, Some(5), Some(-1)]);
    // `3 & 5` is the documented example.
    assert_eq!(i32s(&run("&", &[&l, &r])), [Some(1), None, None, Some(-1)]);
    assert_eq!(i32s(&run("|", &[&l, &r])), [Some(7), None, None, Some(-1)]);
    assert_eq!(i32s(&run("^", &[&l, &r])), [Some(6), None, None, Some(0)]);
    for name in ["&", "|", "^"] {
        let f = bind(name, &[&l, &r]);
        assert_eq!(f.return_type(), &DataType::Int32);
        // Bitwise operators wrap exactly like Java in both ANSI modes.
        let g = bind_with(name, &[&l, &r], &legacy());
        assert_eq!(
            i32s(&g.invoke(&[col(&l), col(&r)], 4).unwrap()),
            i32s(&f.invoke(&[col(&l), col(&r)], 4).unwrap())
        );
    }
}

#[test]
fn bitwise_keeps_the_operand_width() {
    let (a, b) = (i8s(&[Some(3), None]), i8s(&[Some(5), Some(1)]));
    assert_eq!(bind("&", &[&a, &b]).return_type(), &DataType::Int8);
    assert_eq!(i8vals(&run("&", &[&a, &b])), [Some(1), None]);
    let (a, b) = (i16s(&[Some(12)]), i16s(&[Some(10)]));
    assert_eq!(bind("|", &[&a, &b]).return_type(), &DataType::Int16);
    assert_eq!(
        run("|", &[&a, &b])
            .as_primitive::<Int16Type>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(14)]
    );
    let (a, b) = (longs(&[Some(-1), None]), longs(&[Some(1), Some(1)]));
    assert_eq!(bind("^", &[&a, &b]).return_type(), &DataType::Int64);
    assert_eq!(i64vals(&run("^", &[&a, &b])), [Some(-2), None]);
}

#[test]
fn bitwise_not_keeps_the_width() {
    let a = ints(&[Some(0), None, Some(-1)]);
    assert_eq!(i32s(&run("~", &[&a])), [Some(-1), None, Some(0)]);
    let b = longs(&[Some(0)]);
    assert_eq!(bind("~", &[&b]).return_type(), &DataType::Int64);
    assert_eq!(i64vals(&run("~", &[&b])), [Some(-1)]);
    let c = i8s(&[Some(1)]);
    assert_eq!(bind("~", &[&c]).return_type(), &DataType::Int8);
    assert_eq!(i8vals(&run("~", &[&c])), [Some(-2)]);
}

#[test]
fn shifts_match_java_masking() {
    // The `bitwise.sql` goldens.
    for (name, want) in [
        (">>", 5_045_279),
        ("<<", 80_724_468),
        (">>>", 5_045_279),
        ("shiftright", 5_045_279),
        ("shiftleft", 80_724_468),
        ("shiftrightunsigned", 5_045_279),
    ] {
        let base = ints(&[Some(20_181_117)]);
        let by = ints(&[Some(2)]);
        assert_eq!(i32s(&run(name, &[&base, &by])), [Some(want)], "{name}");
    }
    // Negative amounts wrap around the mask.
    assert_eq!(
        i32s(&run("<<", &[&ints(&[Some(1)]), &ints(&[Some(-1)])])),
        [Some(i32::MIN)]
    );
    assert_eq!(
        i32s(&run(">>", &[&ints(&[Some(-8)]), &ints(&[Some(33)])])),
        [Some(-4)]
    );
    // `>>>` is logical: `-8 >>> 2` is `1073741822`.
    assert_eq!(
        i32s(&run(">>>", &[&ints(&[Some(-8)]), &ints(&[Some(2)])])),
        [Some(1_073_741_822)]
    );
    // Longs shift long.
    assert_eq!(
        i64vals(&run("<<", &[&longs(&[Some(1)]), &ints(&[Some(63)])])),
        [Some(i64::MIN)]
    );
    assert_eq!(
        i64vals(&run(">>>", &[&longs(&[Some(-1)]), &ints(&[Some(1)])])),
        [Some(i64::MAX)]
    );
    // NULLs propagate, in both ANSI modes.
    let (a, b) = (ints(&[None]), ints(&[Some(1)]));
    assert_eq!(i32s(&run("<<", &[&a, &b])), [None]);
    let f = bind_with(">>", &[&a, &b], &legacy());
    assert_eq!(i32s(&f.invoke(&[col(&a), col(&b)], 1).unwrap()), [None]);
}

#[test]
fn bit_count_counts_sign_extended_bits() {
    // The `bitwise.sql` goldens.
    for (v, want) in [(Some(true), 1), (Some(false), 0)] {
        let a: ArrayRef = Arc::new(arrow_array::BooleanArray::from(vec![v]));
        assert_eq!(i32s(&run("bit_count", &[&a])), [Some(want)]);
    }
    assert_eq!(
        i32s(&run(
            "bit_count",
            &[&ints(&[Some(1), Some(2), Some(3), None])]
        )),
        [Some(1), Some(1), Some(2), None]
    );
    assert_eq!(
        i32s(&run(
            "bit_count",
            &[&longs(&[Some(-1), Some(i64::MAX), Some(i64::MIN)])]
        )),
        [Some(64), Some(63), Some(1)]
    );
    // Narrow values sign-extend like `Long.bitCount`.
    assert_eq!(i32s(&run("bit_count", &[&i8s(&[Some(-1)])])), [Some(64)]);
    assert_eq!(i32s(&run("bit_count", &[&i16s(&[Some(1)])])), [Some(1)]);
    assert_eq!(
        bind("bit_count", &[&longs(&[Some(0)])]).return_type(),
        &DataType::Int32
    );
}

#[test]
fn bit_get_reads_right_to_left_from_zero() {
    // The `bitwise.sql` goldens: `11` is `1011`.
    let target = longs(&[Some(11)]);
    for (pos, want) in [(3, 1), (2, 0), (1, 1), (0, 1)] {
        let p = ints(&[Some(pos)]);
        assert_eq!(i8vals(&run("bit_get", &[&target, &p])), [Some(want)]);
        assert_eq!(i8vals(&run("getbit", &[&target, &p])), [Some(want)]);
    }
    assert_eq!(
        i8vals(&run("bit_get", &[&target, &ints(&[Some(63)])])),
        [Some(0)]
    );
    // An `int` value has 32 positions.
    let one = ints(&[Some(1)]);
    assert_eq!(
        i8vals(&run("getbit", &[&one, &ints(&[Some(31)])])),
        [Some(0)]
    );
    // NULLs propagate.
    assert_eq!(
        i8vals(&run("bit_get", &[&longs(&[None]), &ints(&[Some(0)])])),
        [None]
    );
    assert_eq!(i8vals(&run("bit_get", &[&target, &ints(&[None])])), [None]);
}

#[test]
fn bit_get_rejects_positions_past_the_width() {
    // `select getbit(11L, -1)` and `(11L, 64)` in `bitwise.sql`.
    for (name, pos, upper) in [
        ("getbit", -1, "64"),
        ("getbit", 64, "64"),
        ("bit_get", 33, "32"),
    ] {
        let (target, p) = if upper == "64" {
            (longs(&[Some(11)]), ints(&[Some(pos)]))
        } else {
            (ints(&[Some(11)]), ints(&[Some(pos)]))
        };
        let f = bind(name, &[&target, &p]);
        assert_eq!(f.return_type(), &DataType::Int8);
        let err = f.invoke(&[col(&target), col(&p)], 1).unwrap_err();
        assert_eq!(
            class(&err),
            ErrorClass::InvalidParameterValueBitPositionRange
        );
        assert_eq!(
            err.to_string(),
            format!(
                "[INVALID_PARAMETER_VALUE.BIT_POSITION_RANGE] The value of parameter(s) `pos` in \
                 `{name}` is invalid: expects an integer value in [0, {upper}), but got {pos}. \
                 SQLSTATE: 22023"
            ),
            "{err}"
        );
    }
}

#[test]
fn bitwise_rejects_other_types() {
    let (s, i) = (super::strs(&[Some("a")]), ints(&[Some(1)]));
    for (name, args) in [
        ("&", vec![Arc::clone(&s), Arc::clone(&i)]),
        ("~", vec![Arc::clone(&s)]),
        ("<<", vec![Arc::clone(&s), Arc::clone(&i)]),
        ("bit_count", vec![Arc::clone(&s)]),
        ("bit_get", vec![s, i]),
    ] {
        let refs: Vec<&ArrayRef> = args.iter().collect();
        let bound: Vec<BindArg> = refs.iter().map(|a| a.data_type().clone().into()).collect();
        let err = bind_scalar(name, &bound, &BindOptions::new()).unwrap_err();
        assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature, "{name}");
    }
}

#[test]
fn more_widths_and_amounts() {
    // `~` keeps every width.
    assert_eq!(
        run("~", &[&i16s(&[Some(1), None])])
            .as_primitive::<Int16Type>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(-2), None]
    );
    // A `bigint` shifted right keeps its sign.
    assert_eq!(
        i64vals(&run(
            "shiftright",
            &[&longs(&[Some(8), None]), &ints(&[Some(1), Some(1)])]
        )),
        [Some(4), None]
    );
    // `getbit` on narrow values checks their own width.
    assert_eq!(
        i8vals(&run("getbit", &[&i8s(&[Some(1)]), &ints(&[Some(7)])])),
        [Some(0)]
    );
    assert_eq!(
        i8vals(&run("getbit", &[&i16s(&[Some(256)]), &ints(&[Some(8)])])),
        [Some(1)]
    );
}
