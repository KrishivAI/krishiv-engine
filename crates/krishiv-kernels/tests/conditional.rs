// SPDX-License-Identifier: Apache-2.0
//! Lazy conditional evaluation: a branch no row takes never runs (so its
//! ANSI errors never surface), every operand sees exactly the rows that
//! reach it, and on inputs that raise nothing the lazy result equals the
//! eager one.

use std::cell::RefCell;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, BooleanArray, Int32Array, StringArray};
use arrow_schema::DataType;
use krishiv_common::error::ErrorClass;
use krishiv_kernels::conditional::{Lazy, Selection, When, and, case_when, coalesce, if_else, or};
use krishiv_kernels::{AnsiError, KernelCtx, KernelError, Operand, integer};
use proptest::prelude::*;

/// A producer reading `column` and recording the rows it was asked for.
fn column<'a>(
    column: &'a ArrayRef,
    seen: &'a RefCell<Vec<Vec<u32>>>,
) -> impl FnMut(&Selection) -> Result<ArrayRef, KernelError> + 'a {
    move |rows| {
        seen.borrow_mut().push(rows.rows().to_vec());
        rows.gather(column.as_ref())
    }
}

fn ints(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(v.to_vec()))
}

fn values(a: &ArrayRef) -> Vec<Option<i32>> {
    a.as_primitive::<Int32Type>().iter().collect()
}

/// `a % b` on the selected rows, ANSI: a zero divisor raises.
fn remainder<'a>(
    a: &'a ArrayRef,
    b: &'a ArrayRef,
) -> impl FnMut(&Selection) -> Result<ArrayRef, KernelError> + 'a {
    move |rows| {
        let (x, y) = (rows.gather(a.as_ref())?, rows.gather(b.as_ref())?);
        let out = integer::remainder::<Int32Type, AnsiError>(
            Operand::Array(x.as_primitive::<Int32Type>()),
            Operand::Array(y.as_primitive::<Int32Type>()),
            &KernelCtx::new(),
        )?;
        Ok(Arc::new(out))
    }
}

/// `b <> 0` on the selected rows.
fn nonzero(b: &ArrayRef) -> impl FnMut(&Selection) -> Result<ArrayRef, KernelError> + '_ {
    move |rows| {
        let y = rows.gather(b.as_ref())?;
        let out: BooleanArray = y
            .as_primitive::<Int32Type>()
            .iter()
            .map(|v| v.map(|v| v != 0))
            .collect();
        Ok(Arc::new(out))
    }
}

fn literal(v: i32) -> impl FnMut(&Selection) -> Result<ArrayRef, KernelError> {
    move |rows| Ok(Arc::new(Int32Array::from(vec![v; rows.len()])))
}

#[test]
fn an_untaken_branch_never_raises() {
    let a = ints(&[Some(10), Some(20), Some(30), Some(40)]);
    let b = ints(&[Some(3), Some(0), Some(7), Some(0)]);
    let all = Selection::all(4).unwrap();
    // CASE WHEN b <> 0 THEN a % b ELSE -1 END: the zero divisors take the ELSE.
    let (mut cond, mut then, mut other) = (nonzero(&b), remainder(&a, &b), literal(-1));
    let mut when = [When {
        condition: &mut cond,
        value: &mut then,
    }];
    let out = case_when(&all, &mut when, Some(&mut other), &DataType::Int32).unwrap();
    assert_eq!(values(&out), [Some(1), Some(-1), Some(2), Some(-1)]);
    // Evaluated eagerly, the same remainder raises.
    let e = remainder(&a, &b)(&all).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::RemainderByZero);
    // IF(b <> 0, a % b, -1) is the same one-branch CASE.
    let (mut cond, mut then, mut other) = (nonzero(&b), remainder(&a, &b), literal(-1));
    let out = if_else(&all, &mut cond, &mut then, &mut other, &DataType::Int32).unwrap();
    assert_eq!(values(&out), [Some(1), Some(-1), Some(2), Some(-1)]);
    // A branch that is taken still raises.
    let (mut cond, mut then) = (literal_true(), remainder(&a, &b));
    let mut when = [When {
        condition: &mut cond,
        value: &mut then,
    }];
    let e = case_when(&all, &mut when, None, &DataType::Int32).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::RemainderByZero);
}

fn literal_true() -> impl FnMut(&Selection) -> Result<ArrayRef, KernelError> {
    |rows| Ok(Arc::new(BooleanArray::from(vec![true; rows.len()])))
}

fn flags(v: &[Option<bool>]) -> ArrayRef {
    Arc::new(BooleanArray::from(v.to_vec()))
}

#[test]
fn case_when_hands_each_operand_exactly_the_rows_that_reach_it() {
    // Row:      0     1      2      3      4
    let c1 = flags(&[Some(true), Some(false), None, Some(false), Some(false)]);
    let c2 = flags(&[Some(false), Some(true), Some(true), Some(false), None]);
    let v1 = ints(&[Some(10), Some(11), Some(12), Some(13), Some(14)]);
    let v2 = ints(&[Some(20), Some(21), None, Some(23), Some(24)]);
    let v3 = ints(&[Some(30), Some(31), Some(32), Some(33), Some(34)]);
    let seen = [(); 5].map(|()| RefCell::new(Vec::new()));
    let (mut p1, mut p2, mut p3, mut p4, mut p5) = (
        column(&c1, &seen[0]),
        column(&v1, &seen[1]),
        column(&c2, &seen[2]),
        column(&v2, &seen[3]),
        column(&v3, &seen[4]),
    );
    let mut when = [
        When {
            condition: &mut p1,
            value: &mut p2,
        },
        When {
            condition: &mut p3,
            value: &mut p4,
        },
    ];
    let out = case_when(
        &Selection::all(5).unwrap(),
        &mut when,
        Some(&mut p5),
        &DataType::Int32,
    )
    .unwrap();
    // Row 0: first branch; row 1 and row 2 (NULL condition): second; rows 3 and 4 (NULL): ELSE.
    assert_eq!(values(&out), [Some(10), Some(21), None, Some(33), Some(34)]);
    let rows: Vec<Vec<Vec<u32>>> = seen.iter().map(|s| s.borrow().clone()).collect();
    assert_eq!(rows[0], [vec![0, 1, 2, 3, 4]]); // first condition: every row
    assert_eq!(rows[1], [vec![0]]); // its value: the row it selected
    assert_eq!(rows[2], [vec![1, 2, 3, 4]]); // second condition: the rest
    assert_eq!(rows[3], [vec![1, 2]]);
    assert_eq!(rows[4], [vec![3, 4]]); // ELSE: what is left
}

#[test]
fn case_when_without_else_is_null_and_stops_early() {
    let all = Selection::all(3).unwrap();
    let mut never_true = |rows: &Selection| -> Result<ArrayRef, KernelError> {
        Ok(Arc::new(BooleanArray::from(vec![false; rows.len()])))
    };
    let called = RefCell::new(0);
    let mut value = |rows: &Selection| -> Result<ArrayRef, KernelError> {
        *called.borrow_mut() += 1;
        literal(1)(rows)
    };
    let mut when = [When {
        condition: &mut never_true,
        value: &mut value,
    }];
    let out = case_when(&all, &mut when, None, &DataType::Int32).unwrap();
    assert_eq!(values(&out), [None, None, None]);
    assert_eq!(*called.borrow(), 0);
    // No branches at all, and no rows at all.
    assert_eq!(
        values(&case_when(&all, &mut [], None, &DataType::Int32).unwrap()),
        [None, None, None]
    );
    let none = Selection::all(0).unwrap();
    let mut t = literal_true();
    let mut v = literal(1);
    let mut when = [When {
        condition: &mut t,
        value: &mut v,
    }];
    assert_eq!(
        case_when(&none, &mut when, None, &DataType::Int32)
            .unwrap()
            .len(),
        0
    );
}

#[test]
fn a_branch_covering_every_row_is_returned_as_is() {
    let all = Selection::all(3).unwrap();
    let (mut t, mut v) = (literal_true(), literal(7));
    let mut when = [When {
        condition: &mut t,
        value: &mut v,
    }];
    let out = case_when(&all, &mut when, None, &DataType::Int32).unwrap();
    assert_eq!(values(&out), [Some(7), Some(7), Some(7)]);
}

#[test]
fn null_branch_values_take_the_result_type() {
    let all = Selection::all(2).unwrap();
    let c = flags(&[Some(true), Some(false)]);
    let seen = RefCell::new(Vec::new());
    let mut cond = column(&c, &seen);
    let mut null_value = |rows: &Selection| -> Result<ArrayRef, KernelError> {
        Ok(Arc::new(arrow_array::NullArray::new(rows.len())))
    };
    let mut other = literal(5);
    let out = if_else(
        &all,
        &mut cond,
        &mut null_value,
        &mut other,
        &DataType::Int32,
    )
    .unwrap();
    assert_eq!(values(&out), [None, Some(5)]);
    // A value of another type is an engine error.
    let mut text = |rows: &Selection| -> Result<ArrayRef, KernelError> {
        Ok(Arc::new(StringArray::from(vec!["a"; rows.len()])))
    };
    let mut cond = column(&c, &seen);
    let e = if_else(&all, &mut cond, &mut text, &mut other, &DataType::Int32).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

#[test]
fn producers_that_break_the_contract_are_engine_errors() {
    let all = Selection::all(3).unwrap();
    let mut short = |_: &Selection| -> Result<ArrayRef, KernelError> { Ok(ints(&[Some(1)])) };
    let mut t = literal_true();
    let mut when = [When {
        condition: &mut t,
        value: &mut short,
    }];
    let e = case_when(&all, &mut when, None, &DataType::Int32).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // A condition that is not boolean.
    let mut not_bool = literal(1);
    let mut v = literal(1);
    let mut when = [When {
        condition: &mut not_bool,
        value: &mut v,
    }];
    assert!(case_when(&all, &mut when, None, &DataType::Int32).is_err());
    // A NULL-typed condition is never true.
    let mut null_cond = |rows: &Selection| -> Result<ArrayRef, KernelError> {
        Ok(Arc::new(arrow_array::NullArray::new(rows.len())))
    };
    let mut v = literal(1);
    let mut other = literal(2);
    let mut when = [When {
        condition: &mut null_cond,
        value: &mut v,
    }];
    let out = case_when(&all, &mut when, Some(&mut other), &DataType::Int32).unwrap();
    assert_eq!(values(&out), [Some(2), Some(2), Some(2)]);
}

#[test]
fn coalesce_evaluates_an_argument_only_where_the_earlier_ones_are_null() {
    let a = ints(&[Some(1), None, None, None]);
    let b = ints(&[Some(9), Some(2), None, None]);
    let c = ints(&[Some(9), Some(9), Some(3), None]);
    let seen = [(); 3].map(|()| RefCell::new(Vec::new()));
    let (mut pa, mut pb, mut pc) = (
        column(&a, &seen[0]),
        column(&b, &seen[1]),
        column(&c, &seen[2]),
    );
    let mut args: [&mut dyn Lazy; 3] = [&mut pa, &mut pb, &mut pc];
    let out = coalesce(&Selection::all(4).unwrap(), &mut args, &DataType::Int32).unwrap();
    assert_eq!(values(&out), [Some(1), Some(2), Some(3), None]);
    assert_eq!(seen[0].borrow().clone(), [vec![0, 1, 2, 3]]);
    assert_eq!(seen[1].borrow().clone(), [vec![1, 2, 3]]);
    assert_eq!(seen[2].borrow().clone(), [vec![2, 3]]);
}

#[test]
fn coalesce_skips_an_error_in_an_argument_no_row_reaches() {
    let a = ints(&[Some(1), Some(2)]);
    let zero = ints(&[Some(0), Some(0)]);
    let seen = RefCell::new(Vec::new());
    let mut first = column(&a, &seen);
    let mut second = remainder(&a, &zero);
    let mut args: [&mut dyn Lazy; 2] = [&mut first, &mut second];
    let out = coalesce(&Selection::all(2).unwrap(), &mut args, &DataType::Int32).unwrap();
    assert_eq!(values(&out), [Some(1), Some(2)]);
    // One row whose first argument is NULL does reach it.
    let a = ints(&[Some(1), None]);
    let mut first = column(&a, &seen);
    let mut second = remainder(&a, &zero);
    let mut args: [&mut dyn Lazy; 2] = [&mut first, &mut second];
    let e = coalesce(&Selection::all(2).unwrap(), &mut args, &DataType::Int32);
    // NULL % 0 is NULL, not an error: NULL rows never raise.
    assert_eq!(values(&e.unwrap()), [Some(1), None]);
    // COALESCE() of nothing is NULL.
    let out = coalesce(&Selection::all(2).unwrap(), &mut [], &DataType::Int32).unwrap();
    assert_eq!(values(&out), [None, None]);
}

#[test]
fn and_or_follow_three_valued_logic_and_short_circuit() {
    let l = [Some(true), Some(false), None];
    let r = [Some(true), Some(false), None];
    let truth = |dominant: bool, a: Option<bool>, b: Option<bool>| match (a, b) {
        (Some(x), _) if x == dominant => Some(dominant),
        (_, Some(y)) if y == dominant => Some(dominant),
        (Some(_), Some(_)) => Some(!dominant),
        _ => None,
    };
    for (name, dominant) in [("and", false), ("or", true)] {
        for &a in &l {
            for &b in &r {
                let lc = flags(&[a]);
                let rc = flags(&[b]);
                let seen = RefCell::new(Vec::new());
                let left_seen = RefCell::new(Vec::new());
                let (mut left, mut right) = (column(&lc, &left_seen), column(&rc, &seen));
                let rows = Selection::all(1).unwrap();
                let got = if dominant {
                    or(&rows, &mut left, &mut right)
                } else {
                    and(&rows, &mut left, &mut right)
                }
                .unwrap();
                assert_eq!(
                    got.iter().collect::<Vec<_>>(),
                    [truth(dominant, a, b)],
                    "{name}({a:?}, {b:?})"
                );
                // The right operand ran iff the left one did not decide the row.
                let ran = !seen.borrow().is_empty();
                assert_eq!(ran, a != Some(dominant), "{name}({a:?}, {b:?})");
            }
        }
    }
}

#[test]
fn and_does_not_evaluate_the_right_side_where_the_left_is_false() {
    // b <> 0 AND a % b = 0: the remainder must not see the zero divisors.
    let a = ints(&[Some(10), Some(7), Some(9)]);
    let b = ints(&[Some(5), Some(0), Some(0)]);
    let mut left = nonzero(&b);
    let mut right = |rows: &Selection| -> Result<ArrayRef, KernelError> {
        let r = remainder(&a, &b)(rows)?;
        let out: BooleanArray = r
            .as_primitive::<Int32Type>()
            .iter()
            .map(|v| v.map(|v| v == 0))
            .collect();
        Ok(Arc::new(out))
    };
    let out = and(&Selection::all(3).unwrap(), &mut left, &mut right).unwrap();
    assert_eq!(
        out.iter().collect::<Vec<_>>(),
        [Some(true), Some(false), Some(false)]
    );
    // The same with the operands swapped raises: the left side runs on every row.
    let mut left = |rows: &Selection| -> Result<ArrayRef, KernelError> {
        let r = remainder(&a, &b)(rows)?;
        let out: BooleanArray = r
            .as_primitive::<Int32Type>()
            .iter()
            .map(|v| v.map(|v| v == 0))
            .collect();
        Ok(Arc::new(out))
    };
    let mut right = nonzero(&b);
    let e = and(&Selection::all(3).unwrap(), &mut left, &mut right).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::RemainderByZero);
}

#[test]
fn nested_conditionals_pass_the_selection_down() {
    // CASE WHEN a > 1 THEN (CASE WHEN b <> 0 THEN a % b ELSE 0 END) ELSE -1 END.
    let a = ints(&[Some(1), Some(5), Some(8), Some(9)]);
    let b = ints(&[Some(0), Some(0), Some(3), Some(4)]);
    let mut outer_cond = |rows: &Selection| -> Result<ArrayRef, KernelError> {
        let x = rows.gather(a.as_ref())?;
        let out: BooleanArray = x
            .as_primitive::<Int32Type>()
            .iter()
            .map(|v| v.map(|v| v > 1))
            .collect();
        Ok(Arc::new(out))
    };
    let mut inner = |rows: &Selection| -> Result<ArrayRef, KernelError> {
        let (mut c, mut t, mut e) = (nonzero(&b), remainder(&a, &b), literal(0));
        let mut when = [When {
            condition: &mut c,
            value: &mut t,
        }];
        case_when(rows, &mut when, Some(&mut e), &DataType::Int32)
    };
    let mut other = literal(-1);
    let mut when = [When {
        condition: &mut outer_cond,
        value: &mut inner,
    }];
    let out = case_when(
        &Selection::all(4).unwrap(),
        &mut when,
        Some(&mut other),
        &DataType::Int32,
    )
    .unwrap();
    assert_eq!(values(&out), [Some(-1), Some(0), Some(2), Some(1)]);
}

#[test]
fn selections_are_validated() {
    assert!(Selection::from_indices(vec![0, 2, 5]).is_ok());
    assert!(Selection::from_indices(vec![]).unwrap().is_empty());
    for bad in [vec![1, 1], vec![3, 2]] {
        let e = Selection::from_indices(bad).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
    }
    assert_eq!(Selection::all(3).unwrap().rows(), [0, 1, 2]);
    assert!(Selection::all(usize::MAX).is_err());
    let col = ints(&[Some(1), Some(2)]);
    assert!(
        Selection::from_indices(vec![5])
            .unwrap()
            .gather(col.as_ref())
            .is_err()
    );
    let picked = Selection::from_indices(vec![1])
        .unwrap()
        .gather(col.as_ref())
        .unwrap();
    assert_eq!(values(&picked), [Some(2)]);
}

fn config() -> ProptestConfig {
    // Miri interprets each case, so it runs two; no failure file (Miri's
    // isolation has no file system).
    let default = ProptestConfig::default();
    ProptestConfig {
        cases: if cfg!(miri) { 2 } else { default.cases },
        failure_persistence: None,
        ..default
    }
}

proptest! {
    #![proptest_config(config())]

    /// On inputs that raise nothing, lazy evaluation equals eager evaluation
    /// followed by a row-wise choice.
    #[test]
    fn lazy_equals_eager_on_error_free_inputs(
        rows in proptest::collection::vec(
            (proptest::option::of(any::<bool>()), proptest::option::of(any::<bool>()),
             proptest::option::of(-1000..1000_i32), proptest::option::of(-1000..1000_i32), proptest::option::of(-1000..1000_i32)),
            0..80,
        )
    ) {
        let n = rows.len();
        let c1 = flags(&rows.iter().map(|r| r.0).collect::<Vec<_>>());
        let c2 = flags(&rows.iter().map(|r| r.1).collect::<Vec<_>>());
        let v1 = ints(&rows.iter().map(|r| r.2).collect::<Vec<_>>());
        let v2 = ints(&rows.iter().map(|r| r.3).collect::<Vec<_>>());
        let v3 = ints(&rows.iter().map(|r| r.4).collect::<Vec<_>>());
        let seen = RefCell::new(Vec::new());
        let (mut p1, mut p2, mut p3, mut p4, mut p5) = (
            column(&c1, &seen), column(&v1, &seen), column(&c2, &seen), column(&v2, &seen), column(&v3, &seen),
        );
        let mut when = [
            When { condition: &mut p1, value: &mut p2 },
            When { condition: &mut p3, value: &mut p4 },
        ];
        let lazy = case_when(&Selection::all(n).unwrap(), &mut when, Some(&mut p5), &DataType::Int32).unwrap();
        let eager: Vec<Option<i32>> = rows
            .iter()
            .map(|r| if r.0 == Some(true) { r.2 } else if r.1 == Some(true) { r.3 } else { r.4 })
            .collect();
        prop_assert_eq!(values(&lazy), eager);
        // COALESCE(v1, v2, v3).
        let (mut a1, mut a2, mut a3) = (column(&v1, &seen), column(&v2, &seen), column(&v3, &seen));
        let mut args: [&mut dyn Lazy; 3] = [&mut a1, &mut a2, &mut a3];
        let got = coalesce(&Selection::all(n).unwrap(), &mut args, &DataType::Int32).unwrap();
        let want: Vec<Option<i32>> = rows.iter().map(|r| r.2.or(r.3).or(r.4)).collect();
        prop_assert_eq!(values(&got), want);
        // c1 AND c2, c1 OR c2.
        let (mut l, mut r) = (column(&c1, &seen), column(&c2, &seen));
        let conj = and(&Selection::all(n).unwrap(), &mut l, &mut r).unwrap();
        let (mut l, mut r) = (column(&c1, &seen), column(&c2, &seen));
        let disj = or(&Selection::all(n).unwrap(), &mut l, &mut r).unwrap();
        for (i, row) in rows.iter().enumerate() {
            let and_want = match (row.0, row.1) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            };
            let or_want = match (row.0, row.1) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            };
            prop_assert_eq!(if conj.is_null(i) { None } else { Some(conj.value(i)) }, and_want);
            prop_assert_eq!(if disj.is_null(i) { None } else { Some(disj.value(i)) }, or_want);
        }
    }
}
