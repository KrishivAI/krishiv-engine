// SPDX-License-Identifier: Apache-2.0
//! Nested literal values (design P3-T012a note, section 5): the checked
//! constructors, equality by bits, the canonical encoding and the display.

use krishiv_types::{ArrayType, DataType, MapType, StructField, StructType};

use crate::{Expr, Literal, LogicalPlan, Plan, StructuralHash};

fn ints(values: &[i32]) -> Literal {
    Literal::array(
        ArrayType::new(DataType::Integer, true),
        values.iter().map(|v| Literal::Integer(*v)).collect(),
    )
    .unwrap()
}

fn hash_of(l: Literal) -> StructuralHash {
    let plan = LogicalPlan::table(["t"]).project(vec![Expr::from(l)]);
    Plan::try_new(plan).unwrap().structural_hash()
}

fn class(e: &krishiv_common::error::KrishivError) -> String {
    e.error_class().name().to_owned()
}

#[test]
fn equal_values_are_equal_and_different_ones_are_not() {
    assert_eq!(ints(&[1, 2]), ints(&[1, 2]));
    assert_ne!(ints(&[1, 2]), ints(&[1, 3]));
    assert_ne!(ints(&[1, 2]), ints(&[1]));
    // The same values under another type differ.
    let longs = Literal::array(
        ArrayType::new(DataType::Long, true),
        vec![Literal::Long(1), Literal::Long(2)],
    )
    .unwrap();
    assert_ne!(ints(&[1, 2]), longs);
    assert_eq!(ints(&[]), ints(&[]));
}

#[test]
fn floats_inside_compare_by_bits() {
    let of = |v: f64| {
        Literal::array(
            ArrayType::new(DataType::Double, false),
            vec![Literal::Double(v)],
        )
        .unwrap()
    };
    assert_eq!(of(f64::NAN), of(f64::NAN));
    assert_ne!(of(0.0), of(-0.0));
    assert_ne!(hash_of(of(0.0)), hash_of(of(-0.0)));
    assert_eq!(hash_of(of(f64::NAN)), hash_of(of(f64::NAN)));
}

#[test]
fn the_hash_tells_values_and_kinds_apart() {
    assert_eq!(hash_of(ints(&[1, 2])), hash_of(ints(&[1, 2])));
    assert_ne!(hash_of(ints(&[1, 2])), hash_of(ints(&[2, 1])));
    let map = Literal::key_value_map(
        MapType::new(DataType::String, DataType::Integer, true),
        vec![(Literal::String("a".to_owned()), Literal::Integer(1))],
    )
    .unwrap();
    let other = Literal::key_value_map(
        MapType::new(DataType::String, DataType::Integer, true),
        vec![(Literal::String("a".to_owned()), Literal::Integer(2))],
    )
    .unwrap();
    assert_ne!(hash_of(map.clone()), hash_of(other));
    assert_ne!(hash_of(map), hash_of(ints(&[1])));
    let st = |v: i32| {
        Literal::struct_value(
            StructType::new([StructField::new("a", DataType::Integer, false)]),
            vec![Literal::Integer(v)],
        )
        .unwrap()
    };
    assert_ne!(hash_of(st(1)), hash_of(st(2)));
}

#[test]
fn nulls_inside_carry_their_type() {
    let ok = Literal::array(
        ArrayType::new(DataType::Integer, true),
        vec![Literal::Integer(1), Literal::Null(Some(DataType::Integer))],
    );
    let Literal::Array {
        elements,
        data_type,
    } = ok.unwrap()
    else {
        panic!("not an array");
    };
    assert_eq!(
        elements,
        vec![Literal::Integer(1), Literal::Null(Some(DataType::Integer))]
    );
    assert_eq!(*data_type, ArrayType::new(DataType::Integer, true));
    // An untyped null, a null of another type, and a null where the type
    // forbids one are refused.
    for (contains_null, element) in [
        (true, Literal::Null(None)),
        (true, Literal::Null(Some(DataType::Long))),
        (false, Literal::Null(Some(DataType::Integer))),
    ] {
        let e = Literal::array(
            ArrayType::new(DataType::Integer, contains_null),
            vec![element],
        )
        .unwrap_err();
        assert_eq!(class(&e), "INTERNAL_ERROR");
    }
}

#[test]
fn the_constructors_check_element_types_and_counts() {
    let e = Literal::array(
        ArrayType::new(DataType::Integer, true),
        vec![Literal::String("x".to_owned())],
    )
    .unwrap_err();
    assert_eq!(class(&e), "INTERNAL_ERROR");
    // A map key is never null.
    let e = Literal::key_value_map(
        MapType::new(DataType::Integer, DataType::Integer, true),
        vec![(Literal::Null(Some(DataType::Integer)), Literal::Integer(1))],
    )
    .unwrap_err();
    assert_eq!(class(&e), "INTERNAL_ERROR");
    // One value per field, a null only in a nullable one.
    let ty = StructType::new([
        StructField::new("a", DataType::Integer, false),
        StructField::new("b", DataType::String, true),
    ]);
    let short = Literal::struct_value(ty.clone(), vec![Literal::Integer(1)]).unwrap_err();
    assert_eq!(class(&short), "INTERNAL_ERROR");
    let null_in_required = Literal::struct_value(
        ty.clone(),
        vec![
            Literal::Null(Some(DataType::Integer)),
            Literal::Null(Some(DataType::String)),
        ],
    )
    .unwrap_err();
    assert_eq!(class(&null_in_required), "INTERNAL_ERROR");
    assert!(
        Literal::struct_value(
            ty,
            vec![Literal::Integer(1), Literal::Null(Some(DataType::String))]
        )
        .is_ok()
    );
}

#[test]
fn the_constructors_check_the_depth() {
    let nest = |depth: usize| {
        let mut t = DataType::Integer;
        for _ in 0..depth {
            t = ArrayType::new(t, true).into();
        }
        t
    };
    let deep = |t: DataType| match t {
        DataType::Array(a) => Literal::array((*a).clone(), Vec::new()),
        _ => unreachable!(),
    };
    assert!(deep(nest(DataType::MAX_NESTING_DEPTH)).is_ok());
    let e = deep(nest(DataType::MAX_NESTING_DEPTH + 1)).unwrap_err();
    assert_eq!(class(&e), "FAILED_TO_PARSE_TOO_COMPLEX");
}

#[test]
fn a_literal_reports_its_type() {
    let l = ints(&[1]);
    assert_eq!(
        l.data_type(),
        Some(ArrayType::new(DataType::Integer, true).into())
    );
    assert_eq!(Literal::Null(None).data_type(), None);
}

#[test]
fn nested_literals_print_as_constructor_calls() {
    // Golden-pending: Spark's text for a nested literal is not established.
    assert_eq!(ints(&[1, 2]).to_string(), "ARRAY(1, 2)");
    let st = Literal::struct_value(
        StructType::new([StructField::new("a", DataType::Integer, false)]),
        vec![Literal::Integer(1)],
    )
    .unwrap();
    assert_eq!(st.to_string(), "NAMED_STRUCT('a', 1)");
    let map = Literal::key_value_map(
        MapType::new(DataType::String, DataType::Integer, true),
        vec![(Literal::String("a".to_owned()), Literal::Integer(1))],
    )
    .unwrap();
    assert_eq!(map.to_string(), "MAP('a', 1)");
}

/// Every pair `Literal::eq` tells apart must encode differently: the
/// canonical type keeps nullability and every struct field.
#[test]
fn the_hash_keeps_nullability_and_every_field_of_the_type() {
    let arr = |contains_null: bool| {
        Literal::array(
            ArrayType::new(DataType::Integer, contains_null),
            vec![Literal::Integer(1)],
        )
        .unwrap()
    };
    assert_ne!(arr(true), arr(false));
    assert_ne!(hash_of(arr(true)), hash_of(arr(false)));

    let map = |value_null: bool| {
        Literal::key_value_map(
            MapType::new(DataType::String, DataType::Integer, value_null),
            vec![(Literal::String("a".to_owned()), Literal::Integer(1))],
        )
        .unwrap()
    };
    assert_ne!(map(true), map(false));
    assert_ne!(hash_of(map(true)), hash_of(map(false)));

    let st = |nullable: bool| {
        Literal::struct_value(
            StructType::new([StructField::new("a", DataType::Integer, nullable)]),
            vec![Literal::Integer(1)],
        )
        .unwrap()
    };
    assert_ne!(st(true), st(false));
    assert_ne!(hash_of(st(true)), hash_of(st(false)));

    // Names that differ only after the 25th field (`simple_string` stops there).
    let wide = |last: &str| {
        let fields: Vec<_> = (0..25)
            .map(|i| StructField::new(format!("f{i}"), DataType::Integer, true))
            .chain([StructField::new(last, DataType::Integer, true)])
            .collect();
        Literal::struct_value(StructType::new(fields), vec![Literal::Integer(1); 26]).unwrap()
    };
    assert_ne!(wide("x"), wide("y"));
    assert_ne!(hash_of(wide("x")), hash_of(wide("y")));
    assert_eq!(hash_of(wide("x")), hash_of(wide("x")));
}

#[test]
fn debug_and_data_type_of_nested_literals() {
    let arr = ints(&[1]);
    let map = Literal::key_value_map(
        MapType::new(DataType::String, DataType::Integer, true),
        vec![(Literal::String("a".to_owned()), Literal::Integer(1))],
    )
    .unwrap();
    let st_type = StructType::new([StructField::new("a", DataType::Integer, false)]);
    let st = Literal::struct_value(st_type.clone(), vec![Literal::Integer(1)]).unwrap();
    assert!(format!("{arr:?}").starts_with("Array"));
    assert!(format!("{map:?}").starts_with("KeyValueMap"));
    assert!(format!("{st:?}").starts_with("Struct"));
    assert_eq!(
        arr.data_type(),
        Some(DataType::Array(
            ArrayType::new(DataType::Integer, true).into()
        ))
    );
    assert!(matches!(map.data_type(), Some(DataType::Map(_))));
    assert_eq!(st.data_type(), Some(DataType::Struct(st_type)));
    let interval = Literal::CalendarInterval {
        months: 1,
        days: 2,
        micros: 3,
    };
    assert!(format!("{interval:?}").starts_with("CalendarInterval"));
    assert_eq!(interval.data_type(), Some(DataType::CalendarInterval));
}

#[test]
fn an_options_only_map_is_not_a_nested_member() {
    let e = Literal::array(
        ArrayType::new(
            DataType::Map(MapType::new(DataType::String, DataType::String, false).into()),
            false,
        ),
        vec![Literal::Map(vec![("k".to_owned(), "v".to_owned())])],
    )
    .unwrap_err();
    assert_eq!(class(&e), "INTERNAL_ERROR");
}

#[test]
fn identity_text_of_an_empty_array_names_its_element_type() {
    let empty = |t: DataType| Expr::from(Literal::array(ArrayType::new(t, true), vec![]).unwrap());
    let (a, b) = (empty(DataType::Integer), empty(DataType::Long));
    assert_eq!(a.to_string(), b.to_string());
    assert_ne!(
        crate::display::identity_string(&a),
        crate::display::identity_string(&b)
    );
}

/// Identity text names the whole nested type, so an empty map of another type, or an array or
/// struct whose type differs only in nullability, is not the same expression (a `GROUP BY` key
/// must not match a select item of another type).
#[test]
fn identity_text_of_nested_literals_names_the_whole_type() {
    let map = |k: DataType, v: DataType| {
        Expr::from(Literal::key_value_map(MapType::new(k, v, true), vec![]).unwrap())
    };
    let (a, b) = (
        map(DataType::String, DataType::Integer),
        map(DataType::String, DataType::Long),
    );
    assert_eq!(a.to_string(), b.to_string());
    assert_ne!(
        crate::display::identity_string(&a),
        crate::display::identity_string(&b)
    );
    let array = |nullable: bool| {
        Expr::from(Literal::array(ArrayType::new(DataType::Integer, nullable), vec![]).unwrap())
    };
    assert_ne!(
        crate::display::identity_string(&array(true)),
        crate::display::identity_string(&array(false))
    );
    let strukt = |nullable: bool| {
        Expr::from(
            Literal::struct_value(
                StructType::new([StructField::new("a", DataType::Integer, nullable)]),
                vec![Literal::Integer(1)],
            )
            .unwrap(),
        )
    };
    assert_ne!(
        crate::display::identity_string(&strukt(true)),
        crate::display::identity_string(&strukt(false))
    );
}
