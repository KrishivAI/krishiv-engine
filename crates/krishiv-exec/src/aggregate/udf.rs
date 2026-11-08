// SPDX-License-Identifier: Apache-2.0
//! Session aggregate UDFs (ADR 0045) in the hash aggregate.
//!
//! An [`AggregateFunction::Udf`](super::AggregateFunction::Udf) names its
//! implementation by name and version ([`UdafCall`]); [`UdafEnv`] is the
//! worker's library it resolves in. [`UdfAcc`] wraps the UDF's
//! [`GroupsAccumulator`] as one accumulator kind, with the engine's side
//! of the contract:
//! - every call goes through [`guard_call`]; an error, a panic or a broken
//!   contract (a state or result of the wrong count, length, type or
//!   nullability) fails the query with `FAILED_EXECUTE_UDF`, and after a
//!   panic in `resize`, `update`, `merge`, `retract` or `size` the
//!   accumulator is never called again (`state` and `evaluate` read only, so
//!   a panic there fails the call and leaves the state as it was);
//! - rows a `FILTER` drops, and with
//!   [`Strict`](krishiv_udf_sdk::NullHandling::Strict) rows with a NULL
//!   argument, never reach `update`;
//! - rows of weight 1 are `update`d, of weight −1 `retract`ed (an
//!   accumulator without retraction is `KRISHIV_UNSUPPORTED_FEATURE`), any
//!   other weight is `KRISHIV_UNSUPPORTED_FEATURE`;
//! - arguments reach the UDF with decimals on `Decimal128`, as scalar UDFs
//!   get them; a `STRING`/`BINARY` result comes back on `Utf8`/`Binary`
//!   and becomes a view;
//! - [`size`](UdfAcc::size) is the UDF's own count, read after every fold;
//!   before a fold the table reserves the size plus the previous fold's
//!   growth ([`projected`](UdfAcc::projected)), so a fold that grows as the
//!   last one did is reserved before it runs and a larger one after (ADR
//!   0045 §4: one batch of overshoot at most).

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BinaryViewArray, BooleanArray, StringViewArray, UInt32Array};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{DataType, Field};
use krishiv_common::error::{KrishivError, KrishivUdfNotAvailable};
use krishiv_functions::scalar::failed_execute;
use krishiv_types::arrow::edge_arrow_type;
use krishiv_types::batch::Weights;
use krishiv_udf_sdk::{
    AggregateUdf, ArgField, GroupsAccumulator, InvokeContext, NullHandling, UdfArg, UdfError,
    UdfLibrary, guard_call,
};

use super::{UdafCall, unsupported};
use crate::carrier::recarried_array;
use crate::error::{Result, internal};

/// The session UDFs a worker runs aggregate UDF calls from.
#[derive(Clone, Debug)]
pub struct UdafEnv {
    /// The library holding the implementations.
    pub library: Arc<UdfLibrary>,
    /// The session zone as a fixed offset in seconds, when it is one (the
    /// UDF's [`InvokeContext::time_zone`]).
    pub time_zone: Option<i32>,
}

/// Checks that `env` holds the aggregate UDF of `a`, with the state it was
/// planned with (other aggregates pass).
///
/// # Errors
/// As [`resolve`] and [`check_state`].
pub(crate) fn check_udaf(a: &super::AggregateExpr, env: Option<&UdafEnv>) -> Result<()> {
    match &a.function {
        super::AggregateFunction::Udf(call) => check_state(&resolve(call, env)?, call),
        _ => Ok(()),
    }
}

/// The accumulator of `a`: an aggregate UDF's from `env`, else the built-in
/// one (the window frames' constructor; a hash table has its own).
///
/// # Errors
/// As [`UdfAcc::new`] or [`Accumulator::new`](super::Accumulator::new).
pub(crate) fn accumulator_of(
    a: &super::AggregateExpr,
    ansi: bool,
    env: Option<&UdafEnv>,
) -> Result<super::Accumulator> {
    match &a.function {
        super::AggregateFunction::Udf(call) => {
            super::Accumulator::udf(a, UdfAcc::new(call, env, ansi)?)
        }
        _ => super::Accumulator::new(a, ansi),
    }
}

impl UdafEnv {
    /// The environment of a session's `udfs` in the zone `time_zone`
    /// (`None` without a library).
    #[must_use]
    pub fn of(udfs: Option<&Arc<UdfLibrary>>, time_zone: Option<i32>) -> Option<Self> {
        udfs.map(|library| Self {
            library: Arc::clone(library),
            time_zone,
        })
    }
}

/// The implementation of `call` in `env`.
///
/// # Errors
/// `KRISHIV_UDF_NOT_AVAILABLE` when the library holds no aggregate UDF of
/// that name and version (or there is no library).
pub(crate) fn resolve(call: &UdafCall, env: Option<&UdafEnv>) -> Result<Arc<dyn AggregateUdf>> {
    let udf = env
        .and_then(|e| e.library.get_aggregate_versioned(&call.name, call.version))
        .ok_or_else(|| {
            KrishivError::from(KrishivUdfNotAvailable {
                name: call.name.clone(),
                version: call.version.to_string(),
            })
        })?;
    // The library lets a panicking `version` through so that binding reports
    // it: report it here, before the implementation runs.
    match guard_call(&call.name, "version", || udf.version()) {
        Ok(v) if v == call.version => Ok(udf),
        Ok(v) => Err(early(
            &call.name,
            &UdfError::new(format!("version {v} where the plan has {}", call.version)),
        )),
        Err(e) => Err(early(&call.name, &e)),
    }
}

/// The argument fields the UDF is typed with: the call's Spark argument
/// types, as the analyzer typed them (`None` for an untyped `NULL`), with
/// the nullability of its columns.
pub(crate) fn arg_fields(call: &UdafCall) -> Result<Vec<ArgField>> {
    if call.args.len() != call.arg_types.len() {
        return Err(internal(format!(
            "aggregate UDF {} has {} argument columns and {} argument types",
            call.name,
            call.args.len(),
            call.arg_types.len()
        )));
    }
    Ok(call
        .args
        .iter()
        .zip(&call.arg_types)
        .map(|(f, t)| ArgField {
            data_type: (*t != krishiv_types::DataType::Null).then(|| t.clone()),
            nullable: f.is_nullable(),
        })
        .collect())
}

/// The carrier a UDF returns `t` on: `Utf8` for `STRING`, `Binary` for
/// `BINARY`, decimals on `Decimal128`, calendar intervals on `Interval(MonthDayNano)` (the session edge's carriers;
/// Spark hands Arrow UDFs a `CalendarInterval` the same way), the canonical type otherwise.
fn plain_carrier(t: &krishiv_types::DataType) -> DataType {
    match t {
        krishiv_types::DataType::String => DataType::Utf8,
        krishiv_types::DataType::Binary => DataType::Binary,
        _ => edge_arrow_type(&t.to_arrow()),
    }
}

/// `FAILED_EXECUTE_UDF` for a call of `name` without a known signature
/// (before the signature could be read).
fn early(name: &str, e: &UdfError) -> KrishivError {
    failed_execute(name, "unknown", "unknown", e.message())
}

/// The partial-state columns `udf` declares for `args`: unprefixed names,
/// the carriers the UDF returns them on.
///
/// # Errors
/// `FAILED_EXECUTE_UDF` when the UDF fails, panics or declares no column.
pub(crate) fn declared_state(
    udf: &Arc<dyn AggregateUdf>,
    name: &str,
    args: &[ArgField],
) -> Result<Vec<Field>> {
    let fields = match guard_call(name, "state_fields", || udf.state_fields(args)) {
        Ok(Ok(f)) => f,
        Ok(Err(e)) | Err(e) => return Err(early(name, &e)),
    };
    if fields.is_empty() {
        return Err(failed_execute(
            name,
            "unknown",
            "unknown",
            "declares no state columns",
        ));
    }
    Ok(fields
        .into_iter()
        .map(|f| Field::new(f.name, plain_carrier(&f.data_type), f.nullable))
        .collect())
}

/// Fills `call.state` from the UDF `env` holds for it (planning).
///
/// # Errors
/// As [`resolve`] and [`declared_state`].
pub(crate) fn plan_udaf_state(call: &mut UdafCall, env: Option<&UdafEnv>) -> Result<()> {
    let udf = resolve(call, env)?;
    call.state = declared_state(&udf, &call.name, &arg_fields(call)?)?;
    Ok(())
}

/// Checks that the worker's `udf` declares the state `call` was planned
/// with (a different implementation under the same version would read
/// states it did not write).
///
/// # Errors
/// `FAILED_EXECUTE_UDF` for a different state, or as [`declared_state`].
pub(crate) fn check_state(udf: &Arc<dyn AggregateUdf>, call: &UdafCall) -> Result<()> {
    let args = arg_fields(call)?;
    let state = declared_state(udf, &call.name, &args)?;
    if state != call.state {
        return Err(failed_execute(
            &call.name,
            "unknown",
            "unknown",
            &format!(
                "declares the state {state:?} where the plan has {:?} for version {}",
                call.state, call.version
            ),
        ));
    }
    Ok(())
}

/// A session aggregate UDF's accumulator, as one accumulator kind.
#[derive(Debug)]
pub(crate) struct UdfAcc {
    /// Behind a lock only so the accumulator is `Sync`, as tables held by
    /// a `Sink` (`Send + Sync`) must be (one driver owns it; the lock is
    /// never contended). A panic is caught inside the guarded call, so it
    /// never poisons it.
    acc: Mutex<Box<dyn GroupsAccumulator>>,
    name: String,
    /// The argument types as Spark prints them (for errors).
    signature: String,
    /// The result type as Spark prints it (for errors).
    result_sql: String,
    /// The carrier the UDF returns its result on.
    out: DataType,
    /// The plan's result type.
    canonical: DataType,
    nullable: bool,
    /// The declared state columns.
    state: Vec<Field>,
    strict: bool,
    /// The groups `resize` created.
    groups: usize,
    /// The UDF's heap bytes after the last fold.
    bytes: usize,
    /// The growth of the fold in progress (its `resize` and fold calls).
    grown: usize,
    /// The growth of the last complete fold.
    growth: usize,
    /// The failure that poisoned the accumulator (a panic, or an error in
    /// a call that cannot report one).
    failed: Option<String>,
}

impl UdfAcc {
    /// A new, empty accumulator of `call` from `env`.
    ///
    /// # Errors
    /// `KRISHIV_UDF_NOT_AVAILABLE` as [`resolve`]; `FAILED_EXECUTE_UDF` when
    /// the UDF fails or panics creating it.
    pub(crate) fn new(call: &UdafCall, env: Option<&UdafEnv>, ansi: bool) -> Result<Self> {
        let udf = resolve(call, env)?;
        let name = call.name.clone();
        let args = arg_fields(call)?;
        let signature = guard_call(&name, "signature", || udf.signature().inputs_sql())
            .map_err(|e| early(&name, &e))?;
        let properties =
            guard_call(&name, "properties", || udf.properties()).map_err(|e| early(&name, &e))?;
        let ctx = InvokeContext {
            partition: 0,
            ansi,
            time_zone: env.and_then(|e| e.time_zone),
        };
        let result_sql = call.result.sql();
        let acc = match guard_call(&name, "accumulator", || udf.accumulator(&args, &ctx)) {
            Ok(Ok(a)) => a,
            Ok(Err(e)) | Err(e) => {
                return Err(failed_execute(&name, &signature, &result_sql, e.message()));
            }
        };
        let mut me = Self {
            acc: Mutex::new(acc),
            name,
            signature,
            result_sql,
            out: plain_carrier(&call.result),
            canonical: call.result.to_arrow(),
            nullable: call.nullable,
            state: call.state.clone(),
            strict: matches!(properties.null_handling, NullHandling::Strict),
            groups: 0,
            bytes: 0,
            grown: 0,
            growth: 0,
            failed: None,
        };
        me.refresh_size();
        Ok(me)
    }

    fn acc(&self) -> MutexGuard<'_, Box<dyn GroupsAccumulator>> {
        self.acc.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `FAILED_EXECUTE_UDF` for this UDF and `reason`.
    fn fail(&self, reason: &str) -> KrishivError {
        failed_execute(&self.name, &self.signature, &self.result_sql, reason)
    }

    /// The poisoning failure, if any.
    fn healthy(&self) -> Result<()> {
        match &self.failed {
            Some(reason) => Err(self.fail(reason)),
            None => Ok(()),
        }
    }

    /// The error of a guarded call: a panic also poisons the accumulator.
    fn caught(&mut self, e: &UdfError, panicked: bool) -> KrishivError {
        if panicked {
            self.failed = Some(e.message().to_owned());
        }
        self.fail(e.message())
    }

    fn refresh_size(&mut self) {
        if self.failed.is_some() {
            return;
        }
        let acc = self.acc();
        let size = guard_call(&self.name, "size", || acc.size());
        drop(acc);
        match size {
            Ok(b) => {
                self.grown = self.grown.saturating_add(b.saturating_sub(self.bytes));
                self.bytes = b;
            }
            Err(e) => self.failed = Some(e.message().to_owned()),
        }
    }

    /// Ends a fold: its growth predicts the next one's.
    fn end_fold(&mut self) {
        self.refresh_size();
        self.growth = std::mem::take(&mut self.grown);
    }

    /// The bytes to reserve before the next fold: the size plus the last
    /// fold's growth.
    pub(crate) const fn projected(&self) -> usize {
        self.bytes.saturating_add(self.growth)
    }

    /// The heap bytes the UDF reported after the last fold.
    pub(crate) const fn size(&self) -> usize {
        self.bytes
    }

    /// The number of state columns.
    pub(crate) fn state_len(&self) -> usize {
        self.state.len()
    }

    /// Makes room for `n` groups. A panic poisons the accumulator; the next
    /// fallible call reports it.
    pub(crate) fn resize(&mut self, n: usize) {
        if self.failed.is_some() || n <= self.groups {
            return;
        }
        let mut acc = self.acc();
        let resized = guard_call(&self.name, "resize", || acc.resize(n));
        drop(acc);
        match resized {
            Ok(()) => self.groups = n,
            Err(e) => self.failed = Some(e.message().to_owned()),
        }
        self.refresh_size();
    }

    /// Folds raw rows: `args` are the argument columns, `mask` the
    /// `FILTER` column, `w` the rows' weights.
    ///
    /// # Errors
    /// `FAILED_EXECUTE_UDF` from the UDF; `KRISHIV_UNSUPPORTED_FEATURE` for
    /// a weight other than ±1, or −1 without retraction.
    pub(crate) fn update(
        &mut self,
        groups: &[u32],
        args: &[&ArrayRef],
        mask: Option<&BooleanArray>,
        w: &Weights,
    ) -> Result<()> {
        self.healthy()?;
        let n = groups.len();
        if args.iter().any(|a| a.len() != n) {
            return Err(internal(format!(
                "aggregate UDF {} folded {n} groups over other argument lengths",
                self.name
            )));
        }
        let kept = self.kept(args, mask, n);
        let mut plus = Vec::new();
        let mut minus = Vec::new();
        for i in 0..n {
            if !kept.as_ref().is_none_or(|k| k.value(i)) {
                continue;
            }
            match weight(w, i) {
                1 => plus.push(i),
                -1 => minus.push(i),
                other => {
                    return Err(unsupported(format!(
                        "an input weight of {other} for the aggregate UDF {}",
                        self.name
                    )));
                }
            }
        }
        let wide: Vec<ArrayRef> = args
            .iter()
            .map(|a| recarried_array(a, &edge_arrow_type(a.data_type())))
            .collect::<std::result::Result<_, _>>()?;
        // TODO(unit 3b / P4 IVM): retracted rows fold after the added ones of
        // the batch; an order-sensitive UDAF would need the batch split at
        // each change of sign. Batch queries only add.
        if plus.len() == n {
            self.call_update(&wide, groups, false)?;
        } else {
            if !plus.is_empty() {
                let (a, g) = take_rows(&wide, groups, &plus)?;
                self.call_update(&a, &g, false)?;
            }
            if !minus.is_empty() {
                let (a, g) = take_rows(&wide, groups, &minus)?;
                self.call_update(&a, &g, true)?;
            }
        }
        self.end_fold();
        self.healthy()
    }

    /// The rows that reach the UDF: those the mask passes and, when
    /// strict, those without a NULL argument (`None`: every row).
    fn kept(
        &self,
        args: &[&ArrayRef],
        mask: Option<&BooleanArray>,
        n: usize,
    ) -> Option<BooleanBuffer> {
        let mut kept: Option<BooleanBuffer> = mask.map(|m| match m.nulls() {
            Some(nulls) => m.values() & nulls.inner(),
            None => m.values().clone(),
        });
        if self.strict {
            for a in args {
                if let Some(nulls) = a.logical_nulls() {
                    let valid = nulls.inner().clone();
                    kept = Some(match kept {
                        None => valid,
                        Some(k) => &k & &valid,
                    });
                }
            }
        }
        kept.filter(|k| k.len() != n || k.count_set_bits() != n)
    }

    fn call_update(&mut self, args: &[ArrayRef], groups: &[u32], retract: bool) -> Result<()> {
        let udf_args: Vec<UdfArg<'_>> = args.iter().map(UdfArg::Column).collect();
        let mut acc = self.acc();
        let out = if retract {
            guard_call(&self.name, "retract", || acc.retract(&udf_args, groups))
        } else {
            guard_call(&self.name, "update", || {
                acc.update(&udf_args, groups).map(|()| true)
            })
        };
        drop(acc);
        match out {
            Ok(Ok(true)) => Ok(()),
            Ok(Ok(false)) => Err(unsupported(format!(
                "retracting rows from the aggregate UDF {}",
                self.name
            ))),
            Ok(Err(e)) => Err(self.caught(&e, false)),
            Err(e) => Err(self.caught(&e, true)),
        }
    }

    /// Folds partial states (one column per declared state field).
    ///
    /// # Errors
    /// `FAILED_EXECUTE_UDF` from the UDF; `INTERNAL_ERROR` for states of
    /// another shape than declared.
    pub(crate) fn merge(&mut self, groups: &[u32], states: &[ArrayRef]) -> Result<()> {
        self.healthy()?;
        if states.len() != self.state.len()
            || states
                .iter()
                .zip(&self.state)
                .any(|(s, f)| s.len() != groups.len() || s.data_type() != f.data_type())
        {
            return Err(internal(format!(
                "aggregate UDF {} merged states of another shape than {:?}",
                self.name, self.state
            )));
        }
        let mut acc = self.acc();
        let merged = guard_call(&self.name, "merge", || acc.merge(states, groups));
        drop(acc);
        match merged {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(self.caught(&e, false)),
            Err(e) => return Err(self.caught(&e, true)),
        }
        self.end_fold();
        self.healthy()
    }

    /// The partial states of `ids`.
    ///
    /// # Errors
    /// `FAILED_EXECUTE_UDF` from the UDF, or for states of the wrong count,
    /// length, type or nullability.
    pub(crate) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        self.healthy()?;
        self.check_ids(ids)?;
        let acc = self.acc();
        let out = match guard_call(&self.name, "state", || acc.state(ids)) {
            Ok(Ok(s)) => s,
            Ok(Err(e)) | Err(e) => return Err(self.fail(e.message())),
        };
        if out.len() != self.state.len() {
            return Err(self.fail(&format!(
                "returned {} state columns where it declares {}",
                out.len(),
                self.state.len()
            )));
        }
        for (s, f) in out.iter().zip(&self.state) {
            self.check(s, ids.len(), f.data_type(), f.is_nullable(), f.name())?;
        }
        Ok(out)
    }

    /// The final values of `ids`, on the plan's result type.
    ///
    /// # Errors
    /// `FAILED_EXECUTE_UDF` from the UDF, or for a result of the wrong
    /// length, type or nullability.
    pub(crate) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        self.healthy()?;
        self.check_ids(ids)?;
        let acc = self.acc();
        let out = match guard_call(&self.name, "evaluate", || acc.evaluate(ids)) {
            Ok(Ok(v)) => v,
            Ok(Err(e)) | Err(e) => return Err(self.fail(e.message())),
        };
        self.check(&out, ids.len(), &self.out, self.nullable, "result")?;
        let out = match (out.data_type(), &self.canonical) {
            (DataType::Utf8, DataType::Utf8View) => {
                Arc::new(StringViewArray::from(out.as_string::<i32>())) as ArrayRef
            }
            (DataType::Binary, DataType::BinaryView) => {
                Arc::new(BinaryViewArray::from(out.as_binary::<i32>()))
            }
            _ => out,
        };
        recarried_array(&out, &self.canonical)
    }

    fn check_ids(&self, ids: &[u32]) -> Result<()> {
        match ids.iter().find(|&&g| g as usize >= self.groups) {
            Some(g) => Err(internal(format!(
                "aggregate UDF {} read group {g} of {}",
                self.name, self.groups
            ))),
            None => Ok(()),
        }
    }

    /// Checks one returned column: `rows` values of `t`, no nulls unless
    /// `nullable`.
    fn check(
        &self,
        a: &ArrayRef,
        rows: usize,
        t: &DataType,
        nullable: bool,
        what: &str,
    ) -> Result<()> {
        if a.len() != rows || a.data_type() != t {
            return Err(self.fail(&format!(
                "returned {} rows of {} as its {what} for {rows} groups of {t}",
                a.len(),
                a.data_type()
            )));
        }
        if !nullable && a.null_count() > 0 {
            return Err(self.fail(&format!(
                "returned {} nulls as its {what}, which it declares non-nullable",
                a.null_count()
            )));
        }
        Ok(())
    }
}

fn weight(w: &Weights, i: usize) -> i64 {
    match w {
        Weights::Const(c) => *c,
        Weights::Column(c) => c.value(i),
    }
}

/// `args` and `groups` at rows `rows`, in order.
fn take_rows(
    args: &[ArrayRef],
    groups: &[u32],
    rows: &[usize],
) -> Result<(Vec<ArrayRef>, Vec<u32>)> {
    let indices: UInt32Array = rows
        .iter()
        .map(|&i| u32::try_from(i).map_err(|_| internal("a batch too large to filter".to_owned())))
        .collect::<Result<_>>()?;
    let taken = args
        .iter()
        .map(|a| {
            arrow_select::take::take(a.as_ref(), &indices, None)
                .map_err(|e| internal(e.to_string()))
        })
        .collect::<Result<_>>()?;
    Ok((taken, rows.iter().map(|&i| groups[i]).collect()))
}

#[cfg(test)]
mod tests;
