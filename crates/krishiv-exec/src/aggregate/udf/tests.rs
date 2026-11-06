// SPDX-License-Identifier: Apache-2.0
//! The engine's side of the aggregate UDF contract, on [`UdfAcc`] directly:
//! resolution and state checks, NULL and FILTER handling, weights, result
//! and state checks, carriers, poisoning and memory projection.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, StringArray};
use arrow_schema::{DataType, Field};
use krishiv_common::error::ErrorClass;
use krishiv_types::batch::Weights;
use krishiv_udf_sdk::{
    AggregateProperties, AggregateUdf, ArgField, GroupsAccumulator, InvokeContext, NullHandling,
    ReturnField, Signature, StateField, UdfArg, UdfError, UdfLibrary,
};

use super::*;

/// What [`Probe`] does beyond counting the rows it is given per group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Count,
    Text,
    NullResult,
    Panic,
    Retract,
    WideState,
}

#[derive(Debug)]
struct Probe {
    mode: Mode,
    nulls: NullHandling,
    signature: Signature,
}

impl AggregateUdf for Probe {
    fn name(&self) -> &str {
        "probe"
    }
    fn version(&self) -> u32 {
        1
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
        Ok(ReturnField {
            data_type: krishiv_types::DataType::Long,
            nullable: false,
        })
    }
    fn state_fields(&self, _: &[ArgField]) -> Result<Vec<StateField>, UdfError> {
        let mut f = vec![StateField {
            name: "rows".to_owned(),
            data_type: krishiv_types::DataType::Long,
            nullable: false,
        }];
        if self.mode == Mode::WideState {
            f.push(f[0].clone());
        }
        Ok(f)
    }
    fn properties(&self) -> AggregateProperties {
        AggregateProperties::DEFAULT.with_null_handling(self.nulls)
    }
    fn accumulator(
        &self,
        _: &[ArgField],
        _: &InvokeContext,
    ) -> Result<Box<dyn GroupsAccumulator>, UdfError> {
        Ok(Box::new(Rows {
            mode: self.mode,
            rows: Vec::new(),
        }))
    }
}

#[derive(Debug)]
struct Rows {
    mode: Mode,
    rows: Vec<i64>,
}

impl GroupsAccumulator for Rows {
    fn resize(&mut self, groups: usize) {
        self.rows.resize(groups, 0);
    }
    fn update(&mut self, _: &[UdfArg<'_>], groups: &[u32]) -> Result<(), UdfError> {
        assert!(self.mode != Mode::Panic, "probe panics");
        for &g in groups {
            self.rows[g as usize] += 1;
        }
        Ok(())
    }
    fn retract(&mut self, _: &[UdfArg<'_>], groups: &[u32]) -> Result<bool, UdfError> {
        if self.mode != Mode::Retract {
            return Ok(false);
        }
        for &g in groups {
            self.rows[g as usize] -= 1;
        }
        Ok(true)
    }
    fn merge(&mut self, states: &[ArrayRef], groups: &[u32]) -> Result<(), UdfError> {
        let s = states[0].as_primitive::<Int64Type>();
        for (i, &g) in groups.iter().enumerate() {
            self.rows[g as usize] += s.value(i);
        }
        Ok(())
    }
    fn state(&self, groups: &[u32]) -> Result<Vec<ArrayRef>, UdfError> {
        // `WideState` declares two columns but returns one.
        Ok(vec![self.counts(groups)])
    }
    fn evaluate(&self, groups: &[u32]) -> Result<ArrayRef, UdfError> {
        Ok(match self.mode {
            Mode::Text => Arc::new(StringArray::from_iter_values(
                groups.iter().map(|&g| self.rows[g as usize].to_string()),
            )),
            Mode::NullResult => Arc::new(Int64Array::from(vec![None; groups.len()])),
            _ => self.counts(groups),
        })
    }
    fn size(&self) -> usize {
        self.rows.len() * 8
    }
}

impl Rows {
    fn counts(&self, groups: &[u32]) -> ArrayRef {
        Arc::new(Int64Array::from_iter_values(
            groups.iter().map(|&g| self.rows[g as usize]),
        ))
    }
}

fn env(mode: Mode, nulls: NullHandling) -> UdafEnv {
    let library = UdfLibrary::new();
    library
        .register_aggregate(Arc::new(Probe {
            mode,
            nulls,
            signature: Signature::Exact(Vec::new()),
        }))
        .unwrap();
    UdafEnv {
        library: Arc::new(library),
        time_zone: None,
    }
}

fn call(result: krishiv_types::DataType) -> UdafCall {
    UdafCall {
        name: "probe".to_owned(),
        version: 1,
        args: vec![
            Field::new("arg0", DataType::Int64, true),
            Field::new("arg1", DataType::Int64, true),
        ],
        result,
        nullable: false,
        state: vec![Field::new("rows", DataType::Int64, false)],
    }
}

fn acc(mode: Mode, nulls: NullHandling) -> UdfAcc {
    let e = env(mode, nulls);
    let mut a = UdfAcc::new(&call(krishiv_types::DataType::Long), Some(&e), true).unwrap();
    a.resize(2);
    a
}

fn col(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn counts(a: &UdfAcc) -> Vec<i64> {
    a.evaluate(&[0, 1])
        .unwrap()
        .as_primitive::<Int64Type>()
        .values()
        .to_vec()
}

#[test]
fn a_missing_udf_or_another_state_refuses_the_plan() {
    let c = call(krishiv_types::DataType::Long);
    let e = resolve(&c, None).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUdfNotAvailable);
    let other = UdafCall {
        version: 2,
        ..c.clone()
    };
    let lib = env(Mode::Count, NullHandling::Strict);
    let e = resolve(&other, Some(&lib)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUdfNotAvailable);
    // Same name and version, another state: a different implementation.
    let wide = env(Mode::WideState, NullHandling::Strict);
    let udf = resolve(&c, Some(&wide)).unwrap();
    let e = check_state(&udf, &c).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedExecuteUdf);
    check_state(&resolve(&c, Some(&lib)).unwrap(), &c).unwrap();
}

#[test]
fn strict_udfs_never_see_a_null_in_any_argument_nor_a_filtered_row() {
    let mut a = acc(Mode::Count, NullHandling::Strict);
    let x = col(&[Some(1), None, Some(3), Some(4), Some(5)]);
    let y = col(&[Some(1), Some(2), None, Some(4), Some(5)]);
    // The mask drops row 3 (false) and row 4 (NULL).
    let mask = BooleanArray::from(vec![Some(true), Some(true), Some(true), Some(false), None]);
    a.update(&[0, 0, 1, 1, 1], &[&x, &y], Some(&mask), &Weights::Const(1))
        .unwrap();
    assert_eq!(counts(&a), [1, 0]);
    let mut c = acc(Mode::Count, NullHandling::Custom);
    c.update(&[0, 0, 1, 1, 1], &[&x, &y], Some(&mask), &Weights::Const(1))
        .unwrap();
    assert_eq!(counts(&c), [2, 1]);
}

#[test]
fn weights_retract_or_are_refused() {
    let x = col(&[Some(1), Some(2), Some(3)]);
    let minus = Weights::Column(Int64Array::from(vec![1, -1, 1]));
    let mut r = acc(Mode::Retract, NullHandling::Strict);
    r.update(&[0, 1, 1], &[&x], None, &minus).unwrap();
    assert_eq!(counts(&r), [1, 0]);
    let mut n = acc(Mode::Count, NullHandling::Strict);
    let e = n.update(&[0, 1, 1], &[&x], None, &minus).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    let two = Weights::Column(Int64Array::from(vec![1, 2, 1]));
    let e = r.update(&[0, 1, 1], &[&x], None, &two).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn results_and_states_are_checked_against_the_declaration() {
    let x = col(&[Some(1)]);
    let mut a = acc(Mode::NullResult, NullHandling::Strict);
    a.update(&[0], &[&x], None, &Weights::Const(1)).unwrap();
    let e = a.evaluate(&[0]).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedExecuteUdf);
    assert!(e.message().contains("non-nullable"), "{}", e.message());
    // The plan declares one state column; the UDF returns a wrong one.
    let wide = env(Mode::WideState, NullHandling::Strict);
    let mut two = call(krishiv_types::DataType::Long);
    two.state.push(Field::new("rows", DataType::Int64, false));
    let mut w = UdfAcc::new(&two, Some(&wide), true).unwrap();
    w.resize(1);
    let e = w.state(&[0]).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedExecuteUdf);
    // Merging states of another shape is an engine fault.
    let e = acc(Mode::Count, NullHandling::Strict)
        .merge(&[0], &[col(&[Some(1)]), col(&[Some(1)])])
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // A group the UDF never created.
    let e = acc(Mode::Count, NullHandling::Strict)
        .evaluate(&[5])
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

#[test]
fn a_string_result_becomes_a_view() {
    let e = env(Mode::Text, NullHandling::Strict);
    let mut a = UdfAcc::new(&call(krishiv_types::DataType::String), Some(&e), true).unwrap();
    a.resize(1);
    a.update(
        &[0, 0],
        &[&col(&[Some(1), Some(2)])],
        None,
        &Weights::Const(1),
    )
    .unwrap();
    let out = a.evaluate(&[0]).unwrap();
    assert_eq!(out.data_type(), &DataType::Utf8View);
    assert_eq!(out.as_string_view().value(0), "2");
}

#[test]
fn a_panic_in_a_fold_poisons_the_accumulator() {
    let mut a = acc(Mode::Panic, NullHandling::Strict);
    let x = col(&[Some(1)]);
    let e = a.update(&[0], &[&x], None, &Weights::Const(1)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedExecuteUdf);
    assert!(e.message().contains("probe panics"), "{}", e.message());
    let e = a.evaluate(&[0]).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::FailedExecuteUdf);
}

#[test]
fn the_projection_reserves_the_last_folds_growth() {
    let mut a = acc(Mode::Count, NullHandling::Strict);
    let x = col(&[Some(1)]);
    a.update(&[0], &[&x], None, &Weights::Const(1)).unwrap();
    assert_eq!(a.size(), 16);
    a.resize(10);
    a.update(&[9], &[&x], None, &Weights::Const(1)).unwrap();
    // 8 groups (64 bytes) were added by that fold: the next is projected alike.
    assert_eq!(a.size(), 80);
    assert_eq!(a.projected(), 144);
}
