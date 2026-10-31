// SPDX-License-Identifier: Apache-2.0
//! `IN (list)`, `CASE WHEN`, `coalesce` (lowered to `CASE WHEN` so untaken
//! branches are never evaluated) and the single-evaluation conditionals
//! `nullif` / `nullifzero` (`NullIf`) and `nanvl` (`Nanvl`). Every other
//! conditional (`if`, `nvl`, `nvl2`, `ifnull`, `zeroifnull`) is rewritten to
//! these in the analyzer, which owns all type coercion: the nodes here take
//! the analyzer's casts and derive no types.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, new_null_array};
use arrow_buffer::BooleanBuffer;
use arrow_schema::DataType as ArrowType;
use arrow_select::filter::filter_record_batch;
use arrow_select::interleave::interleave;
use arrow_select::zip::zip;
use krishiv_kernels::KernelCtx;
use krishiv_kernels::compare::CompareKind;
use krishiv_plan::{Builtin, CastMode, ResolvedFunction};

use super::cast::{CastParts, apply, canonical_carrier, parts};
use super::logic::{cmp, compare, is_null};
use super::{EvalEnv, Input, PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::internal;

/// `v IN (a, b, ...)` with `v` evaluated once: true on a match, NULL
/// when `v` is NULL or nothing matched and an element is NULL, else
/// false. Operands of one Spark `STRING`/`BINARY` in different Arrow
/// encodings meet in the canonical carrier (each side cast once at plan
/// time); every other mismatch fails here exactly as `=` does, through
/// [`compare`](super::logic::compare) validation.
pub(super) fn in_list(
    mut args: Vec<Arc<dyn PhysExpr>>,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let list = args.split_off(1);
    let mut v = args.pop().ok_or_else(|| internal("IN arity"))?;
    if list.is_empty() {
        return Err(internal("IN without a list"));
    }
    let mut items = list;
    if let Some(target) =
        string_target(std::iter::once(v.data_type()).chain(items.iter().map(|e| e.data_type())))
    {
        if v.data_type() != &target && *v.data_type() != ArrowType::Null {
            v = super::cast::coerce_to(v, &target, env)?;
        }
        for item in &mut items {
            if item.data_type() != &target && *item.data_type() != ArrowType::Null {
                *item = super::cast::coerce_to(Arc::clone(item), &target, env)?;
            }
        }
    }
    // The same gate `=` applies, for the same error on other mismatches.
    for item in &items {
        compare(Builtin::Eq, Arc::clone(&v), Arc::clone(item), env)?;
    }
    if items.len() > IN_SET_THRESHOLD
        && items.iter().all(|e| e.is_constant())
        && let Some(set) = InSet::try_new(&v, &items)
    {
        return Ok(Arc::new(set));
    }
    Ok(Arc::new(InList {
        v,
        items,
        kernels: env.kernels,
    }))
}

/// The canonical carrier when every non-`NULL` type is one Spark
/// `STRING`/`BINARY` in some Arrow encoding: `None` otherwise (including
/// all-`NULL`, for which no cast is needed).
fn string_target<'a>(types: impl IntoIterator<Item = &'a ArrowType>) -> Option<ArrowType> {
    let mut out: Option<ArrowType> = None;
    for t in types {
        if *t == ArrowType::Null {
            continue;
        }
        let c = canonical_carrier(t)?;
        match &out {
            None => out = Some(c),
            Some(a) if *a == c => {}
            _ => return None,
        }
    }
    out
}

/// More constant list elements than this make an `IN` a hash-set lookup
/// (Spark's `spark.sql.optimizer.inSetConversionThreshold` default, above
/// which its optimizer rewrites `In` to `InSet`).
const IN_SET_THRESHOLD: usize = 10;

/// `v IN (constants)` as a set lookup (Spark's `InSet`): one hash probe
/// per row instead of one comparison per element. The same answers as
/// [`InList`]: true on a match, NULL when `v` is NULL or nothing matched
/// and the list holds a NULL, else false. Built only for types whose `=`
/// is equality of one canonical value: strings and binaries by their
/// bytes; integers, dates, timestamps and booleans as `i64`; decimals as
/// `i128` (one precision and scale, the analyzer's common type). Floats
/// (`-0.0 = 0.0`, NaN) keep [`InList`].
#[derive(Debug)]
struct InSet {
    v: Arc<dyn PhysExpr>,
    keys: Keys,
    /// The list holds a NULL.
    has_null: bool,
}

/// A hash set with a fixed-seed fast hasher (plan data must not depend on
/// a per-process seed; the set is only probed, never iterated).
type FastSet<T> = std::collections::HashSet<T, foldhash::fast::FixedState>;

fn fast_set<T>(n: usize) -> FastSet<T> {
    FastSet::with_capacity_and_hasher(n, foldhash::fast::FixedState::with_seed(0))
}

/// How [`InSet`] keys its values.
enum Keys {
    /// String or binary bytes.
    Bytes(FastSet<Vec<u8>>),
    /// Integers, dates, timestamps and booleans.
    Ints(FastSet<i64>),
    /// Decimals (unscaled).
    Decimals(FastSet<i128>),
}

impl std::fmt::Debug for Keys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes(s) => write!(f, "Bytes({} keys)", s.len()),
            Self::Ints(s) => write!(f, "Ints({} keys)", s.len()),
            Self::Decimals(s) => write!(f, "Decimals({} keys)", s.len()),
        }
    }
}

/// The bytes of row `i` of a string or binary array.
fn byte_value(a: &dyn Array, i: usize) -> &[u8] {
    match a.data_type() {
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => {
            super::logic::str_value(a, i).as_bytes()
        }
        _ => super::logic::bin_value(a, i),
    }
}

/// Calls `f(row, value)` for every row of an integer-like array (`None`
/// for a NULL row), one loop per Arrow type; `false` if `a` is not one.
fn for_each_int(a: &dyn Array, mut f: impl FnMut(Option<i64>)) -> bool {
    use arrow_array::types::{
        Date32Type, Int8Type, Int16Type, Int32Type, Int64Type, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
    };
    macro_rules! each {
        ($t:ty) => {
            a.as_primitive::<$t>()
                .iter()
                .for_each(|v| f(v.map(i64::from)))
        };
    }
    match a.data_type() {
        ArrowType::Int8 => each!(Int8Type),
        ArrowType::Int16 => each!(Int16Type),
        ArrowType::Int32 => each!(Int32Type),
        ArrowType::Int64 => each!(Int64Type),
        ArrowType::Date32 => each!(Date32Type),
        ArrowType::Timestamp(arrow_schema::TimeUnit::Second, _) => each!(TimestampSecondType),
        ArrowType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => {
            each!(TimestampMillisecondType);
        }
        ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            each!(TimestampMicrosecondType);
        }
        ArrowType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => {
            each!(TimestampNanosecondType);
        }
        ArrowType::Boolean => a.as_boolean().iter().for_each(|v| f(v.map(i64::from))),
        _ => return false,
    }
    true
}

/// Whether [`for_each_int`] reads arrays of type `t`.
fn int_keyed(t: &ArrowType) -> bool {
    matches!(
        t,
        ArrowType::Int8
            | ArrowType::Int16
            | ArrowType::Int32
            | ArrowType::Int64
            | ArrowType::Date32
            | ArrowType::Timestamp(..)
            | ArrowType::Boolean
    )
}

impl InSet {
    /// The set of `items` (constants) for `v`, or `None` when `v`'s type
    /// has no exact key or an element does not evaluate here (it then
    /// keeps the per-row [`InList`], which raises any error only when a
    /// row reaches it, as Spark does).
    fn try_new(v: &Arc<dyn PhysExpr>, items: &[Arc<dyn PhysExpr>]) -> Option<Self> {
        let t = v.data_type().clone();
        let empty = RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty()));
        let mut has_null = false;
        let mut values: Vec<ArrayRef> = Vec::with_capacity(items.len());
        for item in items {
            let a = Arc::clone(item.eval(&empty, 0).ok()?.array());
            if a.len() != 1 {
                return None;
            }
            if a.is_null(0) || *a.data_type() == ArrowType::Null {
                has_null = true;
                continue;
            }
            if *a.data_type() != t {
                return None;
            }
            values.push(a);
        }
        let keys = match t {
            ArrowType::Utf8
            | ArrowType::LargeUtf8
            | ArrowType::Utf8View
            | ArrowType::Binary
            | ArrowType::LargeBinary
            | ArrowType::BinaryView => {
                let mut set = fast_set(values.len());
                set.extend(values.iter().map(|a| byte_value(a.as_ref(), 0).to_vec()));
                Keys::Bytes(set)
            }
            ArrowType::Decimal128(..) => {
                let mut set = fast_set(values.len());
                set.extend(values.iter().map(|a| {
                    a.as_primitive::<arrow_array::types::Decimal128Type>()
                        .value(0)
                }));
                Keys::Decimals(set)
            }
            _ => {
                // With every element NULL no element has been checked against
                // the probe's type: only an integer-like probe has an exact key.
                if !int_keyed(&t) {
                    return None;
                }
                let mut set = fast_set(values.len());
                for a in &values {
                    let ok = for_each_int(a.as_ref(), |v| {
                        if let Some(v) = v {
                            set.insert(v);
                        }
                    });
                    if !ok {
                        return None;
                    }
                }
                Keys::Ints(set)
            }
        };
        Some(Self {
            v: Arc::clone(v),
            keys,
            has_null,
        })
    }
}

impl PhysExpr for InSet {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        self.v.nullable() || self.has_null
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.v.eval(batch, partition)?;
        let a = v.array();
        let n = a.len();
        let mut values = Vec::with_capacity(n);
        let mut valid = Vec::with_capacity(n);
        let mut decide = |found: Option<bool>| match found {
            // NULL `v`.
            None => {
                values.push(false);
                valid.push(false);
            }
            Some(true) => {
                values.push(true);
                valid.push(true);
            }
            Some(false) => {
                values.push(false);
                valid.push(!self.has_null);
            }
        };
        match &self.keys {
            Keys::Bytes(set) => {
                for i in 0..n {
                    decide(
                        a.is_valid(i)
                            .then(|| set.contains(byte_value(a.as_ref(), i))),
                    );
                }
            }
            Keys::Decimals(set) => {
                let d = a.as_primitive::<arrow_array::types::Decimal128Type>();
                d.iter().for_each(|x| decide(x.map(|x| set.contains(&x))));
            }
            Keys::Ints(set) => {
                if !for_each_int(a.as_ref(), |x| decide(x.map(|x| set.contains(&x)))) {
                    return Err(internal("IN set over a value of another type"));
                }
            }
        }
        let out: ArrayRef = Arc::new(BooleanArray::new(
            BooleanBuffer::from(values),
            Some(arrow_buffer::NullBuffer::new(BooleanBuffer::from(valid))),
        ));
        Ok(if v.is_scalar() {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}

/// `IN (list)` over one evaluation of `v`: each list element is evaluated
/// at most once, and only until every row has matched (further elements
/// cannot change a `true`). This reuses the evaluated canonical `Value`
/// of `v` across the elements instead of converting it once per element.
#[derive(Debug)]
struct InList {
    v: Arc<dyn PhysExpr>,
    items: Vec<Arc<dyn PhysExpr>>,
    kernels: KernelCtx,
}

impl PhysExpr for InList {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        self.v.nullable() || self.items.iter().any(|e| e.nullable())
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.v.eval(batch, partition)?;
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(Value::Array(Arc::new(BooleanArray::new(
                BooleanBuffer::new_unset(0),
                None,
            ))));
        }
        let mut matched = vec![false; rows];
        let mut saw_null = vec![false; rows];
        let mut saw_array = !v.is_scalar();
        for item in &self.items {
            if matched.iter().all(|m| *m) {
                break;
            }
            let other = item.eval(batch, partition)?;
            saw_array = saw_array || !other.is_scalar();
            let eq = cmp(CompareKind::Equal, &v, &other, &self.kernels)?;
            // `eq` is one row for scalar-vs-scalar, else `rows`.
            let scalar_eq = eq.len() == 1 && rows != 1;
            for i in 0..rows {
                if matched[i] {
                    continue;
                }
                let at = if scalar_eq { 0 } else { i };
                if !eq.is_valid(at) {
                    saw_null[i] = true;
                } else if eq.value(at) {
                    matched[i] = true;
                }
            }
        }
        if !saw_array {
            let m = matched.first().copied().unwrap_or(false);
            let s = saw_null.first().copied().unwrap_or(false);
            let out: ArrayRef = Arc::new(BooleanArray::new(
                BooleanBuffer::from(vec![m]),
                (!m && s).then(|| arrow_buffer::NullBuffer::new(BooleanBuffer::from(vec![false]))),
            ));
            return Ok(Value::Scalar(out));
        }
        let mut values = vec![false; rows];
        let mut valid = vec![true; rows];
        for i in 0..rows {
            if matched[i] {
                values[i] = true;
            } else if saw_null[i] {
                valid[i] = false;
            }
        }
        Ok(Value::Array(Arc::new(BooleanArray::new(
            BooleanBuffer::from(values),
            Some(arrow_buffer::NullBuffer::new(BooleanBuffer::from(valid))),
        ))))
    }
}

/// `v BETWEEN lo AND hi` with `v` evaluated once: `v >= lo AND v <= hi`
/// under Kleene logic. Operands of one Spark `STRING`/`BINARY` in
/// different encodings meet in the canonical carrier; every other
/// mismatch fails here exactly as the two comparisons do.
pub(super) fn between(
    v: Arc<dyn PhysExpr>,
    lo: Arc<dyn PhysExpr>,
    hi: Arc<dyn PhysExpr>,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let (mut v, mut lo, mut hi) = (v, lo, hi);
    if let Some(target) = string_target([v.data_type(), lo.data_type(), hi.data_type()]) {
        for e in [&mut v, &mut lo, &mut hi] {
            if e.data_type() != &target && *e.data_type() != ArrowType::Null {
                *e = super::cast::coerce_to(Arc::clone(e), &target, env)?;
            }
        }
    }
    compare(Builtin::GtEq, Arc::clone(&v), Arc::clone(&lo), env)?;
    compare(Builtin::LtEq, Arc::clone(&v), Arc::clone(&hi), env)?;
    Ok(Arc::new(Between {
        v,
        lo,
        hi,
        kernels: env.kernels,
    }))
}

/// `BETWEEN` over one evaluation of `v`: both bounds reuse the evaluated
/// canonical `Value` instead of converting it once per bound. `hi` is
/// evaluated only where `v >= lo` did not already decide `false`, as the
/// `AND` it lowers to would.
#[derive(Debug)]
struct Between {
    v: Arc<dyn PhysExpr>,
    lo: Arc<dyn PhysExpr>,
    hi: Arc<dyn PhysExpr>,
    kernels: KernelCtx,
}

impl PhysExpr for Between {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        self.v.nullable() || self.lo.nullable() || self.hi.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.v.eval(batch, partition)?;
        let lo = self.lo.eval(batch, partition)?;
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(Value::Array(Arc::new(BooleanArray::new(
                BooleanBuffer::new_unset(0),
                None,
            ))));
        }
        let ge = cmp(CompareKind::GreaterEq, &v, &lo, &self.kernels)?;
        let scalar_ge = ge.len() == 1 && rows != 1;
        // `false AND _` is `false`: skip `hi` where every row decided.
        let all_false = (0..ge.len()).all(|i| ge.is_valid(i) && !ge.value(i));
        if all_false {
            if ge.len() == 1 && v.is_scalar() && lo.is_scalar() {
                let out: ArrayRef =
                    Arc::new(BooleanArray::new(BooleanBuffer::from(vec![false]), None));
                return Ok(Value::Scalar(out));
            }
            let out: ArrayRef = if scalar_ge {
                Arc::new(BooleanArray::new(
                    BooleanBuffer::from(vec![false; rows]),
                    None,
                ))
            } else {
                Arc::new(BooleanArray::new(
                    BooleanBuffer::from(vec![false; ge.len()]),
                    None,
                ))
            };
            return Ok(Value::Array(out));
        }
        let hi = self.hi.eval(batch, partition)?;
        let all_scalar = v.is_scalar() && lo.is_scalar() && hi.is_scalar();
        let le = cmp(CompareKind::LessEq, &v, &hi, &self.kernels)?;
        let n = if all_scalar { 1 } else { rows };
        let mut values = Vec::with_capacity(n);
        let mut valids = Vec::with_capacity(n);
        for i in 0..n {
            let gi = if ge.len() == 1 { 0 } else { i };
            let li = if le.len() == 1 { 0 } else { i };
            let (g_valid, l_valid) = (ge.is_valid(gi), le.is_valid(li));
            let (g, l) = (g_valid && ge.value(gi), l_valid && le.value(li));
            if (g_valid && !g) || (l_valid && !l) {
                values.push(false);
                valids.push(true);
            } else if g_valid && l_valid {
                values.push(true);
                valids.push(true);
            } else {
                values.push(false);
                valids.push(false);
            }
        }
        let out: ArrayRef = Arc::new(BooleanArray::new(
            BooleanBuffer::from(values),
            Some(arrow_buffer::NullBuffer::new(BooleanBuffer::from(valids))),
        ));
        if all_scalar {
            return Ok(Value::Scalar(out));
        }
        if out.len() == 1 && rows != 1 {
            let mut values = Vec::with_capacity(rows);
            let mut valids = Vec::with_capacity(rows);
            for _ in 0..rows {
                values.push(out.as_boolean().value(0));
                valids.push(out.is_valid(0));
            }
            return Ok(Value::Array(Arc::new(BooleanArray::new(
                BooleanBuffer::from(values),
                Some(arrow_buffer::NullBuffer::new(BooleanBuffer::from(valids))),
            ))));
        }
        Ok(Value::Array(out))
    }
}

/// `CASE WHEN c1 THEN v1 ... [ELSE e] END`, lazily: each condition sees only
/// the rows no earlier branch took, each value only the rows it takes.
#[derive(Debug)]
struct When {
    branches: Vec<(Arc<dyn PhysExpr>, Arc<dyn PhysExpr>)>,
    otherwise: Option<Arc<dyn PhysExpr>>,
    out: ArrowType,
}

pub(super) fn when(
    mut args: Vec<Arc<dyn PhysExpr>>,
    out: ArrowType,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let mut otherwise = if args.len() % 2 == 1 {
        args.pop()
    } else {
        None
    };
    let mut branches = Vec::with_capacity(args.len() / 2);
    let mut it = args.into_iter();
    while let (Some(c), Some(v)) = (it.next(), it.next()) {
        if c.data_type() != &ArrowType::Boolean {
            return Err(internal("a CASE condition is not boolean"));
        }
        branches.push((c, v));
    }
    for v in branches.iter_mut().map(|(_, v)| v).chain(&mut otherwise) {
        coerce_branch(v, &out, env)?;
    }
    Ok(Arc::new(When {
        branches,
        otherwise,
        out,
    }))
}

/// `v` as a value of the `CASE` result type `out`: branches of one Spark
/// `STRING`/`BINARY` in different Arrow encodings meet in the canonical
/// carrier; an untyped `NULL` stays one. Anything else is an engine bug:
/// the analyzer has cast every branch to the common type.
fn coerce_branch(v: &mut Arc<dyn PhysExpr>, out: &ArrowType, env: &EvalEnv) -> Result<()> {
    let t = v.data_type().clone();
    if &t == out || t == ArrowType::Null {
        return Ok(());
    }
    if canonical_carrier(&t) == Some(out.clone()) {
        *v = super::cast::coerce_to(Arc::clone(v), out, env)?;
        return Ok(());
    }
    Err(internal(&format!(
        "a CASE value of type {t} in a CASE of type {out}"
    )))
}

impl PhysExpr for When {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.otherwise.as_ref().is_none_or(|e| e.nullable())
            || self.branches.iter().any(|(_, v)| v.nullable())
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        // `rest`: the rows no branch took yet, as a batch and their row
        // numbers in `batch`.
        let mut rest = batch.clone();
        let mut rest_rows: Vec<usize> = (0..rows).collect();
        let mut parts: Vec<ArrayRef> = Vec::new();
        // Per output row: (part, row within the part).
        let mut pick = vec![(0_usize, 0_usize); rows];
        for (cond, value) in &self.branches {
            if rest_rows.is_empty() {
                break;
            }
            let c = cond.eval(&rest, partition)?.into_array(rest.num_rows())?;
            let c = c
                .as_boolean_opt()
                .ok_or_else(|| internal("CASE condition"))?;
            // NULL is not true.
            let taken = match c.nulls() {
                Some(n) => c.values() & n.inner(),
                None => c.values().clone(),
            };
            if taken.count_set_bits() == 0 {
                continue;
            }
            let mask = BooleanArray::new(taken.clone(), None);
            let chosen = filter_record_batch(&rest, &mask).map_err(|e| internal(&e.to_string()))?;
            let v = cast_null(
                value
                    .eval(&chosen, partition)?
                    .into_array(chosen.num_rows())?,
                &self.out,
            );
            let part = parts.len();
            parts.push(v);
            let mut k = 0;
            for (i, r) in rest_rows.iter().enumerate() {
                if taken.value(i) {
                    pick[*r] = (part, k);
                    k += 1;
                }
            }
            let left = BooleanArray::new(!&taken, None);
            rest = filter_record_batch(&rest, &left).map_err(|e| internal(&e.to_string()))?;
            rest_rows = rest_rows
                .iter()
                .zip(taken.iter())
                .filter_map(|(r, t)| (!t).then_some(*r))
                .collect();
        }
        if !rest_rows.is_empty() {
            let v = match &self.otherwise {
                Some(e) => cast_null(
                    e.eval(&rest, partition)?.into_array(rest.num_rows())?,
                    &self.out,
                ),
                None => new_null_array(&self.out, rest_rows.len()),
            };
            let part = parts.len();
            parts.push(v);
            for (k, r) in rest_rows.iter().enumerate() {
                pick[*r] = (part, k);
            }
        }
        if parts.len() == 1 {
            return Ok(Value::Array(parts.swap_remove(0)));
        }
        let refs: Vec<&dyn Array> = parts.iter().map(AsRef::as_ref).collect();
        if refs.is_empty() {
            return Ok(Value::Array(new_null_array(&self.out, 0)));
        }
        interleave(&refs, &pick)
            .map(Value::Array)
            .map_err(|e| internal(&e.to_string()))
    }
}

/// An untyped NULL value as a NULL of `out`.
fn cast_null(a: ArrayRef, out: &ArrowType) -> ArrayRef {
    if a.data_type() == &ArrowType::Null {
        new_null_array(out, a.len())
    } else {
        a
    }
}

/// Binds `coalesce` to `CASE WHEN` machinery, so an untaken branch is never
/// evaluated: `coalesce(1, 1 / 0)` answers `1` without raising, as Spark
/// does. The analyzer has cast every argument to their common type.
///
/// `f` is the analyzed call (for the result type), `args` its compiled
/// arguments. An argument count the analyzer never produces is an engine bug
/// (`INTERNAL_ERROR`).
pub(super) fn coalesce(
    f: &ResolvedFunction,
    args: Vec<Arc<dyn PhysExpr>>,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let out = f.data_type.to_arrow();
    // The first non-null argument: each condition sees only the rows no
    // earlier argument took, each value only the rows it takes.
    let mut acc: Option<Arc<dyn PhysExpr>> = None;
    for a in args.into_iter().rev() {
        acc = Some(match acc {
            None => a,
            Some(taken) => {
                let cond = is_null(false, Arc::clone(&a));
                when(vec![cond, a, taken], out.clone(), env)?
            }
        });
    }
    acc.ok_or_else(|| internal("coalesce without arguments"))
}

/// `nullif(a, b)` / `nullifzero(x)`: NULL where `a = b`, else `a`, with `a`
/// evaluated once. The analyzer has cast `b` to the type `=` compares at
/// (`other`'s type); operands of one Spark `STRING`/`BINARY` in different
/// encodings meet in the canonical carrier instead. The casts are for the
/// comparison only; the result keeps `a`'s values at the result's carrier.
#[derive(Debug)]
pub(super) struct NullIf {
    value: Arc<dyn PhysExpr>,
    other: Arc<dyn PhysExpr>,
    /// Each side to the comparison type, for the comparison only; `None`
    /// when already there (no copy).
    value_cast: Option<CastParts>,
    other_cast: Option<CastParts>,
    /// `value` to the result type: the analyzer types the result at the
    /// value's Spark type, whose canonical carrier may differ from the
    /// value's own (a `Utf8` column in a `Utf8View` result). `None` when
    /// already there; a different Spark type is left for `zip` to reject
    /// as before.
    result_cast: Option<CastParts>,
    kernels: KernelCtx,
    out: ArrowType,
}

/// Binds `nullif` / `nullifzero`: `value` at its own type, `other` at the
/// type `=` compares at. `f` is the analyzed call (for the result type),
/// `args` its compiled arguments (`value`, then `other`). Anything but two
/// arguments is an engine bug: the analyzer derives the comparison type.
pub(super) fn null_if(
    f: &ResolvedFunction,
    args: Vec<Arc<dyn PhysExpr>>,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    let [value, other] = <[Arc<dyn PhysExpr>; 2]>::try_from(args)
        .map_err(|_| internal("nullif without two arguments"))?;
    let mode = if input.env.ansi {
        CastMode::Ansi
    } else {
        CastMode::Legacy
    };
    // Same-typed comparisons need no cast: `apply` would copy the column.
    let (value_ty, other_ty) = (value.data_type().clone(), other.data_type().clone());
    let out = f.data_type.to_arrow();
    let (value_cast, other_cast) =
        match (canonical_carrier(&value_ty), canonical_carrier(&other_ty)) {
            // One Spark `STRING`/`BINARY` in different encodings: compare in
            // the canonical carrier, keeping the side already there.
            (Some(a), Some(b)) if a == b && value_ty != other_ty => {
                let at = |t: &ArrowType| {
                    if t == &a {
                        Ok(None)
                    } else {
                        parts(t, a.clone(), mode, input.env).map(Some)
                    }
                };
                (at(&value_ty)?, at(&other_ty)?)
            }
            _ if value_ty == other_ty => (None, None),
            _ => (
                Some(parts(&value_ty, other_ty.clone(), mode, input.env)?),
                None,
            ),
        };
    // The result keeps the value's Spark type at the result's carrier.
    let result_cast = if value_ty == out {
        None
    } else if canonical_carrier(&value_ty) == Some(out.clone()) {
        Some(parts(&value_ty, out.clone(), mode, input.env)?)
    } else {
        None
    };
    Ok(Arc::new(NullIf {
        value,
        other,
        value_cast,
        other_cast,
        result_cast,
        kernels: input.env.kernels,
        out,
    }))
}

impl PhysExpr for NullIf {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        true
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        // `value` is evaluated once: the comparison casts its array, the
        // result keeps its values.
        let a = self.value.eval(batch, partition)?;
        let b = self.other.eval(batch, partition)?;
        let scalar = a.is_scalar() && b.is_scalar();
        let n = if scalar { 1 } else { rows };
        let a_full = a.into_array(n)?;
        let b_full = b.into_array(n)?;
        // An untyped NULL never equals (the kernels cannot cast a `Null`
        // array either); otherwise each side meets the comparison type.
        let null_side =
            a_full.data_type() == &ArrowType::Null || b_full.data_type() == &ArrowType::Null;
        let ab = if null_side {
            Arc::clone(&a_full)
        } else if let Some(cast) = &self.value_cast {
            apply(cast, &a_full)?
        } else {
            Arc::clone(&a_full)
        };
        let bb = if null_side {
            Arc::clone(&b_full)
        } else if let Some(cast) = &self.other_cast {
            apply(cast, &b_full)?
        } else {
            Arc::clone(&b_full)
        };
        let eq = if null_side {
            BooleanArray::new_null(n)
        } else {
            cmp(
                CompareKind::Equal,
                &Value::Array(ab),
                &Value::Array(bb),
                &self.kernels,
            )?
        };
        // NULL is not equal: keep `value` where the comparison is not true.
        let valid = eq
            .nulls()
            .map(|ns| ns.inner().clone())
            .unwrap_or_else(|| BooleanBuffer::new_set(n));
        let mask = BooleanArray::new(eq.values() & &valid, None);
        let kept = match &self.result_cast {
            Some(cast) => apply(cast, &a_full)?,
            None => Arc::clone(&a_full),
        };
        let out = zip(&mask, &new_null_array(&self.out, n), &kept)
            .map_err(|e| internal(&e.to_string()))?;
        Ok(if scalar {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}

/// `nanvl(a, b)`: `b` where `a` is NaN, else `a`, with `a` evaluated once and
/// `b` only where `a` is NaN. Both sides are already the analyzer's
/// `float`/`double`.
#[derive(Debug)]
struct Nanvl {
    a: Arc<dyn PhysExpr>,
    b: Arc<dyn PhysExpr>,
    out: ArrowType,
}

/// Binds `nanvl` over two same-typed `float`/`double` children. Anything
/// else is an engine bug: the analyzer casts both sides.
pub(super) fn nan_vl(
    f: &ResolvedFunction,
    args: Vec<Arc<dyn PhysExpr>>,
    _input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    let [a, b] = <[Arc<dyn PhysExpr>; 2]>::try_from(args)
        .map_err(|_| internal("nanvl without two arguments"))?;
    let t = a.data_type();
    if t != b.data_type() || !matches!(t, ArrowType::Float32 | ArrowType::Float64) {
        return Err(internal("nanvl of non-float arguments"));
    }
    Ok(Arc::new(Nanvl {
        a,
        b,
        out: f.data_type.to_arrow(),
    }))
}

impl PhysExpr for Nanvl {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.a.nullable() || self.b.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        // `a` is evaluated once; NULL is not NaN.
        let a = self.a.eval(batch, partition)?.into_array(rows)?;
        let nan = match a.data_type() {
            ArrowType::Float32 => BooleanBuffer::from_iter(
                a.as_primitive::<Float32Type>()
                    .iter()
                    .map(|v| v.is_some_and(f32::is_nan)),
            ),
            ArrowType::Float64 => BooleanBuffer::from_iter(
                a.as_primitive::<Float64Type>()
                    .iter()
                    .map(|v| v.is_some_and(f64::is_nan)),
            ),
            t => return Err(internal(&format!("nanvl of {t}"))),
        };
        if nan.count_set_bits() == 0 {
            return Ok(Value::Array(a));
        }
        // `b` sees only the NaN rows, so `nanvl(1, 1 / 0)` never raises.
        let b = super::eval_where(self.b.as_ref(), batch, &nan, partition)?;
        let out =
            zip(&BooleanArray::new(nan, None), &b, &a).map_err(|e| internal(&e.to_string()))?;
        Ok(Value::Array(out))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::types::{Decimal128Type, Float32Type, Float64Type, Int32Type, Int64Type};
    use arrow_array::{
        Decimal128Array, Float64Array, Int32Array, Int64Array, NullArray, StringArray,
        StringViewArray,
    };
    use arrow_schema::{Field, Schema};
    use krishiv_common::config::Config;
    use krishiv_common::error::ErrorClass;
    use krishiv_plan::{AttrId, FunctionId, FunctionKind, Literal};
    use krishiv_types::{DataType, DecimalType};

    use super::*;

    /// [`EvalEnv`] with `spark.sql.ansi.enabled = ansi`.
    fn env(ansi: bool) -> EvalEnv {
        let mut config = Config::new();
        config
            .set(
                "spark.sql.ansi.enabled",
                if ansi { "true" } else { "false" },
            )
            .unwrap();
        EvalEnv::new(&config, Arc::from("test"))
    }

    /// [`Input`] over `schema` with `layout` columns.
    fn input<'a>(
        env: &'a EvalEnv,
        layout: &'a [AttrId],
        schema: &'a std::sync::Arc<Schema>,
    ) -> Input<'a> {
        Input {
            layout,
            schema,
            env,
            subqueries: &[],
        }
    }

    /// An analyzed call of `name` with result type `data_type`.
    fn resolved(name: &str, data_type: DataType) -> ResolvedFunction {
        ResolvedFunction {
            id: FunctionId::Registry {
                name: name.to_owned(),
                id: 0,
            },
            kind: FunctionKind::Scalar,
            args: Vec::new(),
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type,
            nullable: true,
            deterministic: true,
        }
    }

    /// A literal expression.
    fn lit(l: &Literal) -> Arc<dyn PhysExpr> {
        Arc::new(super::super::Lit::new(l).unwrap())
    }

    /// An input-column expression.
    fn column(index: usize, data_type: ArrowType, nullable: bool) -> Arc<dyn PhysExpr> {
        Arc::new(super::super::Column {
            index,
            data_type,
            nullable,
        })
    }

    /// A `rows`-row batch with no usable columns, for literal-only plans.
    fn rows_batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("d", ArrowType::Null, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(NullArray::new(rows))]).unwrap()
    }

    /// Evaluates `e` over `batch` as a column of `batch.num_rows()` rows.
    fn values(e: &Arc<dyn PhysExpr>, batch: &RecordBatch) -> ArrayRef {
        e.eval(batch, 0)
            .unwrap()
            .into_array(batch.num_rows())
            .unwrap()
    }

    /// `1 / 0`: `NULL` without ANSI, `DIVIDE_BY_ZERO` with it.
    fn one_over_zero(env: &EvalEnv) -> Arc<dyn PhysExpr> {
        super::super::arith::binary(
            Builtin::Divide,
            lit(&Literal::Double(1.0)),
            lit(&Literal::Double(0.0)),
            ArrowType::Float64,
            true,
            env,
        )
        .unwrap()
    }

    /// More than ten constant elements that are all NULL remove every key:
    /// a probe type with no exact key (a float) keeps the per-element `IN`,
    /// whose answer is NULL, instead of failing in the set lookup; the
    /// integer-like probes still use the set and answer the same.
    #[test]
    fn an_all_null_long_in_list_keeps_in_list_answers_for_every_probe_type() {
        let env = env(false);
        let nulls =
            || -> Vec<Arc<dyn PhysExpr>> { (0..11).map(|_| lit(&Literal::Null(None))).collect() };
        let floats = Arc::new(Float64Array::from(vec![Some(1.5), None, Some(2.5)]));
        let ints = Arc::new(Int64Array::from(vec![Some(1), None, Some(2)]));
        let strs = Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("f", ArrowType::Float64, true),
            Field::new("i", ArrowType::Int64, true),
            Field::new("s", ArrowType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![floats, ints, strs]).unwrap();
        for (index, t, is_set) in [
            (0, ArrowType::Float64, false),
            (1, ArrowType::Int64, true),
            (2, ArrowType::Utf8, true),
        ] {
            let v = column(index, t, true);
            let mut args = vec![Arc::clone(&v)];
            args.extend(nulls());
            let got = in_list(args, &env).unwrap();
            assert_eq!(format!("{got:?}").starts_with("InSet"), is_set, "{got:?}");
            let list: Arc<dyn PhysExpr> = Arc::new(InList {
                v,
                items: nulls(),
                kernels: env.kernels,
            });
            let (got, want) = (values(&got, &batch), values(&list, &batch));
            assert_eq!(got.as_boolean(), want.as_boolean());
            // `x IN (NULL, ...)` is NULL for every row.
            assert_eq!(got.as_boolean().null_count(), 3);
        }
    }

    /// More than ten constant elements make a set lookup with the same
    /// answers as the per-element `IN`: integers, strings (view and plain),
    /// dates, with and without a NULL element, over a column with NULLs and
    /// over a constant value; a float `IN` keeps the per-element path.
    #[test]
    fn a_long_constant_in_list_is_a_set_with_in_list_answers() {
        let env = env(false);
        let check = |v: Arc<dyn PhysExpr>, items: Vec<Arc<dyn PhysExpr>>, batch: &RecordBatch| {
            let mut args = vec![Arc::clone(&v)];
            args.extend(items.iter().cloned());
            let set = in_list(args, &env).unwrap();
            assert!(format!("{set:?}").starts_with("InSet"), "{set:?}");
            let list: Arc<dyn PhysExpr> = Arc::new(InList {
                v,
                items,
                kernels: env.kernels,
            });
            assert_eq!(set.nullable(), list.nullable());
            let (got, want) = (values(&set, batch), values(&list, batch));
            assert_eq!(got.as_boolean(), want.as_boolean());
        };
        let ints = Arc::new(Int64Array::from(vec![
            Some(3),
            None,
            Some(40),
            Some(7),
            Some(12),
        ]));
        let strs = Arc::new(StringViewArray::from(vec![
            Some("k3"),
            None,
            Some("a value longer than twelve bytes"),
            Some("k40"),
            Some(""),
        ]));
        let plain = Arc::new(StringArray::from(vec![
            Some("k3"),
            Some("x"),
            None,
            Some("k7"),
            Some("k12"),
        ]));
        let dates = Arc::new(arrow_array::Date32Array::from(vec![
            Some(3),
            Some(5),
            None,
            Some(11),
            Some(100),
        ]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", ArrowType::Int64, true),
            Field::new("s", ArrowType::Utf8View, true),
            Field::new("p", ArrowType::Utf8, true),
            Field::new("d", ArrowType::Date32, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![ints, strs, plain, dates]).unwrap();
        for with_null in [false, true] {
            let null = with_null.then(|| lit(&Literal::Null(None)));
            let ints: Vec<_> = (0..12)
                .map(|i| lit(&Literal::Long(i)))
                .chain(null.clone())
                .collect();
            check(column(0, ArrowType::Int64, true), ints.clone(), &batch);
            check(lit(&Literal::Long(5)), ints, &batch);
            let mut strs: Vec<_> = (0..12)
                .map(|i| lit(&Literal::String(format!("k{i}"))))
                .chain(null.clone())
                .collect();
            strs.push(lit(&Literal::String(
                "a value longer than twelve bytes".to_owned(),
            )));
            check(column(1, ArrowType::Utf8View, true), strs.clone(), &batch);
            check(column(2, ArrowType::Utf8, true), strs, &batch);
            let dates: Vec<_> = (0..12)
                .map(|i| lit(&Literal::Date(i)))
                .chain(null)
                .collect();
            check(column(3, ArrowType::Date32, true), dates, &batch);
        }
        // Decimal and zoned timestamp columns.
        let dec = DecimalType::try_new(10, 2).unwrap();
        let darr = Arc::new(
            arrow_array::Decimal128Array::from(vec![Some(150), None, Some(1_234), Some(7)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ) as ArrayRef;
        // A zoned timestamp column, of the literals' own Arrow type.
        let t_type = lit(&Literal::Timestamp(0)).data_type().clone();
        assert!(
            matches!(t_type, ArrowType::Timestamp(_, Some(_))),
            "{t_type}"
        );
        let tarr = arrow_array::make_array(
            arrow_array::TimestampMicrosecondArray::from(vec![Some(3), Some(4), None, Some(99)])
                .into_data()
                .into_builder()
                .data_type(t_type.clone())
                .build()
                .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", darr.data_type().clone(), true),
            Field::new("t", t_type.clone(), true),
        ]));
        let dt = RecordBatch::try_new(schema, vec![darr, tarr]).unwrap();
        let decs: Vec<_> = (0..12)
            .map(|i| {
                lit(&Literal::Decimal {
                    unscaled: i * 150,
                    data_type: dec,
                })
            })
            .collect();
        check(column(0, dt.column(0).data_type().clone(), true), decs, &dt);
        let ts: Vec<_> = (0..12).map(|i| lit(&Literal::Timestamp(i))).collect();
        check(column(1, t_type, true), ts, &dt);
        // As the analyzer coerces them: a BIGINT value against INT
        // literals cast to BIGINT, and a DECIMAL(10,2) value against
        // DECIMAL(3,1) literals cast up, still make a set.
        let schema = Arc::new(Schema::new(vec![Field::new("d", ArrowType::Null, true)]));
        let inp = input(&env, &[], &schema);
        let cast = |e: Arc<dyn PhysExpr>, to: &DataType| {
            super::super::cast::compile(e, to, krishiv_plan::CastMode::Legacy, inp).unwrap()
        };
        let small = DecimalType::try_new(3, 1).unwrap();
        let ints: Vec<_> = (0..12)
            .map(|i| cast(lit(&Literal::Integer(i)), &DataType::Long))
            .collect();
        assert!(ints.iter().all(|e| e.is_constant()));
        check(column(0, ArrowType::Int64, true), ints, &batch);
        let decs: Vec<_> = (0..12)
            .map(|i| {
                cast(
                    lit(&Literal::Decimal {
                        unscaled: i128::from(i) * 15,
                        data_type: small,
                    }),
                    &DataType::Decimal(dec),
                )
            })
            .collect();
        check(column(0, dt.column(0).data_type().clone(), true), decs, &dt);
        let floats: Vec<_> = std::iter::once(column(0, ArrowType::Float64, true))
            .chain((0..12).map(|i| lit(&Literal::Double(f64::from(i)))))
            .collect();
        let e = in_list(floats, &env).unwrap();
        assert!(format!("{e:?}").starts_with("InList"), "{e:?}");
        // Ten elements stay a list, as in Spark.
        let ten: Vec<_> = std::iter::once(column(0, ArrowType::Int64, true))
            .chain((0..10).map(|i| lit(&Literal::Long(i))))
            .collect();
        assert!(format!("{:?}", in_list(ten, &env).unwrap()).starts_with("InList"));
    }

    /// `coalesce` answers the first non-null argument of each row.
    #[test]
    fn coalesce_takes_the_first_non_null_argument_per_row() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", ArrowType::Int32, true),
            Field::new("b", ArrowType::Int32, true),
            Field::new("c", ArrowType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, None, None])),
                Arc::new(Int32Array::from(vec![Some(2), Some(3), None, None])),
                Arc::new(Int32Array::from(vec![Some(4), Some(5), Some(6), None])),
            ],
        )
        .unwrap();
        let f = resolved("coalesce", DataType::Integer);
        let env = env(true);
        let e = coalesce(
            &f,
            vec![
                column(0, ArrowType::Int32, true),
                column(1, ArrowType::Int32, true),
            ],
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(1), Some(3), None, None]),
        );
        // A single argument is itself.
        let one = coalesce(&f, vec![column(0, ArrowType::Int32, true)], &env).unwrap();
        assert_eq!(
            values(&one, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(1), None, None, None]),
        );
        // No arguments is an engine bug: the analyzer always has some.
        assert!(coalesce(&f, Vec::new(), &env).is_err());
    }

    /// `coalesce(1, 1 / 0)` answers without raising; `coalesce(1 / 0, 1)`
    /// raises with ANSI on and answers `1` without.
    #[test]
    fn coalesce_is_lazy_in_both_modes() {
        for ansi in [true, false] {
            let env = env(ansi);
            let batch = rows_batch(2);
            let f = resolved("coalesce", DataType::Double);
            let e = coalesce(
                &f,
                vec![lit(&Literal::Double(1.0)), one_over_zero(&env)],
                &env,
            )
            .unwrap();
            assert_eq!(
                values(&e, &batch).as_primitive::<Float64Type>(),
                &Float64Array::from(vec![Some(1.0), Some(1.0)]),
                "ansi = {ansi}"
            );
            let e = coalesce(
                &f,
                vec![one_over_zero(&env), lit(&Literal::Double(1.0))],
                &env,
            )
            .unwrap();
            if ansi {
                let err = e.eval(&batch, 0).unwrap_err();
                assert_eq!(err.error_class(), ErrorClass::DivideByZero);
            } else {
                assert_eq!(
                    values(&e, &batch).as_primitive::<Float64Type>(),
                    &Float64Array::from(vec![Some(1.0), Some(1.0)]),
                );
            }
        }
    }

    /// `nullif` of one type: `NULL` where equal, the value elsewhere.
    #[test]
    fn nullif_same_type() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            ArrowType::Utf8View,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringViewArray::from(vec![
                Some("x"),
                Some("y"),
                None,
            ]))],
        )
        .unwrap();
        let layout = [AttrId(0)];
        let f = resolved("nullif", DataType::String);
        let e = null_if(
            &f,
            vec![
                column(0, ArrowType::Utf8View, true),
                lit(&Literal::String("x".to_owned())),
            ],
            input(&env, &layout, &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch)
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            [None, Some("y"), None]
        );
        // Two arguments are required: the analyzer always passes both.
        assert!(
            null_if(
                &f,
                vec![lit(&Literal::Integer(1))],
                input(&env, &layout, &schema),
            )
            .is_err()
        );
    }

    /// `nullif(1, 2.1)`: the `INT` value meets the `DOUBLE` other at
    /// `DOUBLE` for the comparison, but the result keeps the `INT` value.
    #[test]
    fn nullif_compares_at_the_other_type_and_keeps_its_own() {
        let env = env(true);
        let batch = rows_batch(1);
        let schema = Arc::new(Schema::empty());
        let f = resolved("nullif", DataType::Integer);
        let e = null_if(
            &f,
            vec![lit(&Literal::Integer(1)), lit(&Literal::Double(2.1))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(1)]),
        );
        let e = null_if(
            &f,
            vec![lit(&Literal::Integer(1)), lit(&Literal::Double(1.0))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![None]),
        );
    }

    /// `nullif(NULL, 1)` is `NULL` and `nullif(1, NULL)` is `1`: an untyped
    /// `NULL` side never equals, so the value is kept.
    #[test]
    fn nullif_with_an_untyped_null_side() {
        let env = env(true);
        let batch = rows_batch(1);
        let schema = Arc::new(Schema::empty());
        let f = resolved("nullif", DataType::Null);
        let e = null_if(
            &f,
            vec![lit(&Literal::Null(None)), lit(&Literal::Integer(1))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(values(&e, &batch).data_type(), &ArrowType::Null);
        let g = resolved("nullif", DataType::Integer);
        let e = null_if(
            &g,
            vec![lit(&Literal::Integer(1)), lit(&Literal::Null(None))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![Some(1)]),
        );
    }

    /// `nullif` casts its value for the comparison in the session's mode:
    /// `nullifzero('abc')` raises an ANSI cast error and answers the
    /// argument without.
    #[test]
    fn nullif_casts_for_the_comparison_only() {
        for (ansi, want) in [(true, None), (false, Some("abc"))] {
            let env = env(ansi);
            let batch = rows_batch(1);
            let schema = Arc::new(Schema::empty());
            let f = resolved("nullif", DataType::String);
            let e = null_if(
                &f,
                vec![
                    lit(&Literal::String("abc".to_owned())),
                    lit(&Literal::Long(0)),
                ],
                input(&env, &[], &schema),
            )
            .unwrap();
            match want {
                None => {
                    let err = e.eval(&batch, 0).unwrap_err();
                    assert_eq!(err.error_class(), ErrorClass::CastInvalidInput);
                    assert_eq!(err.sql_state(), Some("22018"));
                }
                Some(w) => assert_eq!(
                    values(&e, &batch)
                        .as_string_view()
                        .iter()
                        .collect::<Vec<_>>(),
                    [Some(w)],
                ),
            }
        }
        // A `NULL` value casts to nothing: the comparison is NULL.
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            ArrowType::Decimal128(1, 1),
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(
                Decimal128Array::from(vec![Some(0)])
                    .with_precision_and_scale(1, 1)
                    .unwrap(),
            )],
        )
        .unwrap();
        let layout = [AttrId(0)];
        let f = resolved(
            "nullif",
            DataType::Decimal(DecimalType::try_new(1, 1).unwrap()),
        );
        let e = null_if(
            &f,
            vec![
                column(0, ArrowType::Decimal128(1, 1), true),
                lit(&Literal::Integer(0)),
            ],
            input(&env, &layout, &schema),
        )
        .unwrap();
        let got = values(&e, &batch);
        assert_eq!(got.data_type(), &ArrowType::Decimal128(1, 1));
        assert_eq!(
            got.as_primitive::<Decimal128Type>(),
            &Decimal128Array::from(vec![None as Option<i128>])
                .with_precision_and_scale(1, 1)
                .unwrap(),
        );
    }

    /// An expression counting its evaluations: `nullif` evaluates its first
    /// argument once per batch, for the comparison and the value alike.
    #[derive(Debug)]
    struct Count {
        inner: Arc<dyn PhysExpr>,
        n: Arc<AtomicUsize>,
    }

    impl PhysExpr for Count {
        fn data_type(&self) -> &ArrowType {
            self.inner.data_type()
        }

        fn nullable(&self) -> bool {
            self.inner.nullable()
        }

        fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
            self.n.fetch_add(1, Ordering::SeqCst);
            self.inner.eval(batch, partition)
        }
    }

    /// The value is computed once even though it is both compared and
    /// returned.
    #[test]
    fn nullif_evaluates_its_first_argument_once() {
        let env = env(true);
        let batch = rows_batch(3);
        let schema = Arc::new(Schema::empty());
        let n = Arc::new(AtomicUsize::new(0));
        let value: Arc<dyn PhysExpr> = Arc::new(Count {
            inner: lit(&Literal::Long(1)),
            n: Arc::clone(&n),
        });
        let f = resolved("nullif", DataType::Long);
        let e = null_if(
            &f,
            vec![value, lit(&Literal::Long(2))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_primitive::<Int64Type>(),
            &Int64Array::from(vec![Some(1), Some(1), Some(1)]),
        );
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    /// `nanvl(a, b)`: `b` where `a` is NaN, else `a`; NULL is not NaN.
    #[test]
    fn nanvl_takes_the_second_argument_for_nan_only() {
        let env = env(true);
        let batch = rows_batch(4);
        let schema = Arc::new(Schema::empty());
        for (ty, a, b) in [
            (
                DataType::Float,
                Literal::Float(f32::NAN),
                Literal::Float(1.0),
            ),
            (
                DataType::Double,
                Literal::Double(f64::NAN),
                Literal::Double(1.0),
            ),
        ] {
            let f = resolved("nanvl", ty);
            let e = nan_vl(&f, vec![lit(&a), lit(&b)], input(&env, &[], &schema)).unwrap();
            assert_eq!(values(&e, &batch).len(), 4);
        }
        let f = resolved("nanvl", DataType::Double);
        for (a, want) in [
            (f64::NAN, 1.0),
            (2.5, 2.5),
            (-0.0, -0.0),
            (f64::INFINITY, f64::INFINITY),
        ] {
            let e = nan_vl(
                &f,
                vec![lit(&Literal::Double(a)), lit(&Literal::Double(1.0))],
                input(&env, &[], &schema),
            )
            .unwrap();
            let got = values(&e, &batch).as_primitive::<Float64Type>().value(0);
            assert_eq!(got.to_bits(), want.to_bits(), "{a} -> {got}, want {want}");
        }
        let g = resolved("nanvl", DataType::Float);
        for (a, want) in [
            (f32::NAN, 1.0),
            (2.5, 2.5),
            (-0.0, -0.0),
            (f32::INFINITY, f32::INFINITY),
        ] {
            let e = nan_vl(
                &g,
                vec![lit(&Literal::Float(a)), lit(&Literal::Float(1.0))],
                input(&env, &[], &schema),
            )
            .unwrap();
            let got = values(&e, &batch).as_primitive::<Float32Type>().value(0);
            assert_eq!(got.to_bits(), want.to_bits(), "{a} -> {got}, want {want}");
        }
        // Anything but two same-typed floats is an engine bug.
        assert!(
            nan_vl(
                &f,
                vec![lit(&Literal::Double(1.0)), lit(&Literal::Float(1.0)),],
                input(&env, &[], &schema),
            )
            .is_err()
        );
        assert!(
            nan_vl(
                &f,
                vec![lit(&Literal::Double(1.0))],
                input(&env, &[], &schema),
            )
            .is_err()
        );
    }

    /// `nanvl` evaluates `b` only where `a` is NaN: `nanvl(1, 1 / 0)`
    /// answers `1` without raising.
    #[test]
    fn nanvl_is_lazy_in_its_second_argument() {
        for ansi in [true, false] {
            let env = env(ansi);
            let batch = rows_batch(2);
            let schema = Arc::new(Schema::empty());
            let f = resolved("nanvl", DataType::Double);
            let e = nan_vl(
                &f,
                vec![lit(&Literal::Double(1.0)), one_over_zero(&env)],
                input(&env, &[], &schema),
            )
            .unwrap();
            assert_eq!(
                values(&e, &batch).as_primitive::<Float64Type>(),
                &Float64Array::from(vec![Some(1.0), Some(1.0)]),
                "ansi = {ansi}"
            );
            let e = nan_vl(
                &f,
                vec![lit(&Literal::Double(f64::NAN)), one_over_zero(&env)],
                input(&env, &[], &schema),
            )
            .unwrap();
            if ansi {
                let err = e.eval(&batch, 0).unwrap_err();
                assert_eq!(err.error_class(), ErrorClass::DivideByZero);
            } else {
                assert!(values(&e, &batch).as_primitive::<Float64Type>().is_null(0));
            }
        }
    }

    /// `coalesce` of one Spark `STRING` in `Utf8` and `Utf8View` carriers:
    /// every branch meets the `CASE` type, keeping its values.
    #[test]
    fn coalesce_unifies_string_encodings() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("u", ArrowType::Utf8, true),
            Field::new("v", ArrowType::Utf8View, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow_array::StringArray::from(vec![None, Some("b")])) as ArrayRef,
                Arc::new(StringViewArray::from(vec![Some("v"), Some("w")])),
            ],
        )
        .unwrap();
        let f = resolved("coalesce", DataType::String);
        for (args, want) in [
            (
                vec![
                    column(0, ArrowType::Utf8, true),
                    column(1, ArrowType::Utf8View, true),
                ],
                [Some("v"), Some("b")],
            ),
            (
                vec![
                    column(1, ArrowType::Utf8View, true),
                    column(0, ArrowType::Utf8, true),
                ],
                [Some("v"), Some("w")],
            ),
        ] {
            let e = coalesce(&f, args, &env).unwrap();
            assert_eq!(e.data_type(), &ArrowType::Utf8View);
            assert_eq!(
                values(&e, &batch)
                    .as_string_view()
                    .iter()
                    .collect::<Vec<_>>(),
                want
            );
        }
    }

    /// `CASE WHEN` with `Utf8` branches in a `STRING` (`Utf8View`) case:
    /// the branches are cast; an `INT` branch still fails.
    #[test]
    fn when_unifies_string_encodings() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("u", ArrowType::Utf8, true),
            Field::new("i", ArrowType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow_array::StringArray::from(vec![Some("a"), None])) as ArrayRef,
                Arc::new(Int32Array::from(vec![Some(1), Some(2)])),
            ],
        )
        .unwrap();
        let e = when(
            vec![
                lit(&Literal::Boolean(true)),
                column(0, ArrowType::Utf8, true),
                lit(&Literal::String("d".to_owned())),
            ],
            ArrowType::Utf8View,
            &env,
        )
        .unwrap();
        assert_eq!(e.data_type(), &ArrowType::Utf8View);
        assert_eq!(
            values(&e, &batch)
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            [Some("a"), None]
        );
        // A branch of another Spark type still fails as before.
        let err = when(
            vec![
                lit(&Literal::Boolean(true)),
                column(1, ArrowType::Int32, true),
            ],
            ArrowType::Utf8View,
            &env,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("a CASE value of type Int32 in a CASE of type Utf8View"),
            "{err}"
        );
    }

    /// `nullif` of `Utf8` and `Utf8View` operands compares in the view and
    /// returns the `STRING` carrier with the value's contents.
    #[test]
    fn nullif_unifies_string_encodings() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("u", ArrowType::Utf8, true),
            Field::new("v", ArrowType::Utf8View, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow_array::StringArray::from(vec![
                    Some("x"),
                    Some("y"),
                    None,
                ])) as ArrayRef,
                Arc::new(StringViewArray::from(vec![Some("x"), Some("z"), Some("q")])),
            ],
        )
        .unwrap();
        let layout = [AttrId(0), AttrId(1)];
        let input = input(&env, &layout, &schema);
        // A `Utf8` value against a `Utf8View` literal: the result is the
        // `STRING` carrier with the value's contents.
        let f = resolved("nullif", DataType::String);
        let e = null_if(
            &f,
            vec![
                column(0, ArrowType::Utf8, true),
                lit(&Literal::String("x".to_owned())),
            ],
            input,
        )
        .unwrap();
        assert_eq!(e.data_type(), &ArrowType::Utf8View);
        assert_eq!(
            values(&e, &batch)
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            [None, Some("y"), None]
        );
        // A `Utf8View` value against a `Utf8` other: the comparison stays in
        // the view, the result keeps the view.
        let e = null_if(
            &f,
            vec![
                column(1, ArrowType::Utf8View, true),
                column(0, ArrowType::Utf8, true),
            ],
            input,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch)
                .as_string_view()
                .iter()
                .collect::<Vec<_>>(),
            [None, Some("z"), Some("q")]
        );
    }

    /// A two-column batch of one Spark `STRING` in both carriers.
    fn strings_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("u", ArrowType::Utf8, true),
            Field::new("v", ArrowType::Utf8View, true),
        ]));
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec![Some("b"), None])) as ArrayRef,
                Arc::new(StringViewArray::from(vec![Some("b"), Some("c")])),
            ],
        )
        .unwrap()
    }

    /// `IN` over a `Utf8` value with `Utf8View` elements (and the reverse):
    /// each side meets the canonical carrier once, keeping its values.
    #[test]
    fn in_list_meets_mixed_encodings_in_the_view() {
        let env = env(true);
        let batch = strings_batch();
        let e = in_list(
            vec![
                column(0, ArrowType::Utf8, true),
                lit(&Literal::String("b".to_owned())),
                lit(&Literal::String("c".to_owned())),
            ],
            &env,
        )
        .unwrap();
        assert_eq!(e.data_type(), &ArrowType::Boolean);
        assert!(e.nullable());
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [Some(true), None]
        );
        // A `Utf8View` value against a `Utf8` element: the element is cast.
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", ArrowType::Utf8View, true),
            Field::new("u", ArrowType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringViewArray::from(vec![Some("b"), Some("c")])),
                Arc::new(StringArray::from(vec![Some("b"), Some("b")])),
            ],
        )
        .unwrap();
        let e = in_list(
            vec![
                column(0, ArrowType::Utf8View, true),
                column(1, ArrowType::Utf8, true),
                lit(&Literal::String("x".to_owned())),
            ],
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [Some(true), Some(false)]
        );
    }

    /// `IN` without a list is an engine bug; operands of different Spark
    /// types fail here exactly as `=` does.
    #[test]
    fn in_list_rejects_an_empty_list_and_other_spark_types() {
        let env = env(true);
        let err = in_list(vec![column(0, ArrowType::Int32, true)], &env).unwrap_err();
        assert!(err.to_string().contains("IN without a list"), "{err}");
        let err = in_list(
            vec![
                column(0, ArrowType::Int32, true),
                lit(&Literal::String("x".to_owned())),
            ],
            &env,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("comparison of Int32 with Utf8View"),
            "{err}"
        );
    }

    /// An `IN` list of untyped `NULL`s never matches, and a `NULL` value
    /// never matches either: every row is NULL.
    #[test]
    fn in_list_with_only_untyped_nulls_never_matches() {
        let env = env(true);
        let batch = strings_batch();
        let e = in_list(
            vec![
                column(0, ArrowType::Utf8, true),
                lit(&Literal::Null(None)),
                lit(&Literal::Null(None)),
            ],
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [None, None]
        );
        let e = in_list(
            vec![
                lit(&Literal::Null(None)),
                lit(&Literal::String("b".to_owned())),
            ],
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &rows_batch(2))
                .as_boolean()
                .iter()
                .collect::<Vec<_>>(),
            [None, None]
        );
    }

    /// `IN` over a batch without rows answers an empty column.
    #[test]
    fn in_list_on_an_empty_batch_answers_empty() {
        let env = env(true);
        let batch = rows_batch(0);
        let e = in_list(
            vec![lit(&Literal::Integer(1)), lit(&Literal::Integer(2))],
            &env,
        )
        .unwrap();
        let got = e.eval(&batch, 0).unwrap().into_array(0).unwrap();
        assert_eq!(got.len(), 0);
    }

    /// `IN` of scalars is one scalar: a match, or NULL when nothing
    /// matched and an element is NULL.
    #[test]
    fn in_list_of_scalars_is_a_scalar() {
        let env = env(true);
        let batch = rows_batch(2);
        let e = in_list(
            vec![lit(&Literal::Integer(1)), lit(&Literal::Integer(1))],
            &env,
        )
        .unwrap();
        match e.eval(&batch, 0).unwrap() {
            Value::Scalar(a) => assert_eq!(a.as_boolean().iter().collect::<Vec<_>>(), [Some(true)]),
            v => panic!("not a scalar: {v:?}"),
        }
        let e = in_list(
            vec![lit(&Literal::Integer(1)), lit(&Literal::Integer(2))],
            &env,
        )
        .unwrap();
        match e.eval(&batch, 0).unwrap() {
            Value::Scalar(a) => {
                assert_eq!(a.as_boolean().iter().collect::<Vec<_>>(), [Some(false)]);
            }
            v => panic!("not a scalar: {v:?}"),
        }
        let e = in_list(
            vec![lit(&Literal::Integer(1)), lit(&Literal::Null(None))],
            &env,
        )
        .unwrap();
        match e.eval(&batch, 0).unwrap() {
            Value::Scalar(a) => {
                assert_eq!(a.as_boolean().iter().collect::<Vec<_>>(), [None]);
            }
            v => panic!("not a scalar: {v:?}"),
        }
    }

    /// The `IN` value is evaluated once per batch, however many elements
    /// the list holds.
    #[test]
    fn in_list_evaluates_its_value_once() {
        let env = env(true);
        let batch = rows_batch(3);
        let n = Arc::new(AtomicUsize::new(0));
        let value: Arc<dyn PhysExpr> = Arc::new(Count {
            inner: lit(&Literal::Long(1)),
            n: Arc::clone(&n),
        });
        let e = in_list(
            vec![value, lit(&Literal::Long(1)), lit(&Literal::Long(2))],
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [Some(true), Some(true), Some(true)]
        );
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    /// Elements past the point where every row matched are never
    /// evaluated: a later element cannot change a `true`.
    #[test]
    fn in_list_skips_elements_after_every_row_matched() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![Field::new("a", ArrowType::Int32, true)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef],
        )
        .unwrap();
        let counted = |l: &Literal, n: &Arc<AtomicUsize>| -> Arc<dyn PhysExpr> {
            Arc::new(Count {
                inner: lit(l),
                n: Arc::clone(n),
            })
        };
        let (n0, n1, n2) = (
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );
        let e = in_list(
            vec![
                column(0, ArrowType::Int32, true),
                counted(&Literal::Integer(1), &n0),
                counted(&Literal::Integer(2), &n1),
                counted(&Literal::Integer(3), &n2),
            ],
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [Some(true), Some(true)]
        );
        assert_eq!(n0.load(Ordering::SeqCst), 1);
        assert_eq!(n1.load(Ordering::SeqCst), 1);
        assert_eq!(n2.load(Ordering::SeqCst), 0);
    }

    /// `BETWEEN` over a `Utf8` value with `Utf8View` bounds (and a
    /// `Utf8View` value with a `Utf8` bound).
    #[test]
    fn between_meets_mixed_encodings_in_the_view() {
        let env = env(true);
        let batch = strings_batch();
        let e = between(
            column(0, ArrowType::Utf8, true),
            lit(&Literal::String("a".to_owned())),
            lit(&Literal::String("c".to_owned())),
            &env,
        )
        .unwrap();
        assert_eq!(e.data_type(), &ArrowType::Boolean);
        assert!(e.nullable());
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [Some(true), None]
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("w", ArrowType::Utf8View, true),
            Field::new("u", ArrowType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringViewArray::from(vec![Some("b"), Some("c")])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
            ],
        )
        .unwrap();
        let e = between(
            column(0, ArrowType::Utf8View, true),
            column(1, ArrowType::Utf8, true),
            lit(&Literal::String("c".to_owned())),
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [Some(true), Some(true)]
        );
    }

    /// `BETWEEN` over operands of different Spark types fails here exactly
    /// as its two comparisons do.
    #[test]
    fn between_with_other_spark_types_fails_like_its_comparisons() {
        let env = env(true);
        let err = between(
            column(0, ArrowType::Int32, true),
            lit(&Literal::String("a".to_owned())),
            lit(&Literal::String("c".to_owned())),
            &env,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("comparison of Int32 with Utf8View"),
            "{err}"
        );
    }

    /// `LargeUtf8` is one more encoding of `STRING`: `IN` and `BETWEEN`
    /// meet it in the canonical carrier like `Utf8`; a dictionary-encoded
    /// string is no encoding of it and fails as `=` does.
    #[test]
    fn large_utf8_meets_the_view_and_dictionaries_do_not() {
        let env = env(true);
        let schema = Arc::new(Schema::new(vec![
            Field::new("l", ArrowType::LargeUtf8, true),
            Field::new(
                "d",
                ArrowType::Dictionary(Box::new(ArrowType::Int32), Box::new(ArrowType::Utf8)),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow_array::LargeStringArray::from(vec![Some("b"), None])) as ArrayRef,
                Arc::new(
                    arrow_array::DictionaryArray::<arrow_array::types::Int32Type>::from_iter([
                        Some("b"),
                        None,
                    ]),
                ),
            ],
        )
        .unwrap();
        let lit_s = |t: &str| lit(&Literal::String(t.to_owned()));
        let e = in_list(
            vec![
                column(0, ArrowType::LargeUtf8, true),
                lit_s("b"),
                lit_s("c"),
            ],
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [Some(true), None]
        );
        let e = between(
            column(0, ArrowType::LargeUtf8, true),
            lit_s("a"),
            lit_s("c"),
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [Some(true), None]
        );
        let dictionary =
            ArrowType::Dictionary(Box::new(ArrowType::Int32), Box::new(ArrowType::Utf8));
        assert!(in_list(vec![column(1, dictionary.clone(), true), lit_s("b")], &env).is_err());
        assert!(between(column(1, dictionary, true), lit_s("a"), lit_s("c"), &env).is_err());
    }

    /// `BETWEEN` over a batch without rows answers an empty column.
    #[test]
    fn between_on_an_empty_batch_answers_empty() {
        let env = env(true);
        let batch = rows_batch(0);
        let e = between(
            lit(&Literal::Integer(1)),
            lit(&Literal::Integer(0)),
            lit(&Literal::Integer(2)),
            &env,
        )
        .unwrap();
        let got = e.eval(&batch, 0).unwrap().into_array(0).unwrap();
        assert_eq!(got.len(), 0);
    }

    /// `false AND _` is `false`: where `v >= lo` already decided every
    /// row, the upper bound is never evaluated.
    #[test]
    fn between_skips_the_upper_bound_where_the_lower_decided_false() {
        let env = env(true);
        let n = Arc::new(AtomicUsize::new(0));
        let hi: Arc<dyn PhysExpr> = Arc::new(Count {
            inner: lit(&Literal::Long(3)),
            n: Arc::clone(&n),
        });
        let e = between(lit(&Literal::Long(1)), lit(&Literal::Long(5)), hi, &env).unwrap();
        match e.eval(&rows_batch(2), 0).unwrap() {
            Value::Scalar(a) => {
                assert_eq!(a.as_boolean().iter().collect::<Vec<_>>(), [Some(false)]);
            }
            v => panic!("not a scalar: {v:?}"),
        }
        assert_eq!(n.load(Ordering::SeqCst), 0);
        // One decided row does not skip the bound for the rest.
        let schema = Arc::new(Schema::new(vec![Field::new("a", ArrowType::Int32, true)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 7])) as ArrayRef],
        )
        .unwrap();
        let n = Arc::new(AtomicUsize::new(0));
        let hi: Arc<dyn PhysExpr> = Arc::new(Count {
            inner: lit(&Literal::Integer(6)),
            n: Arc::clone(&n),
        });
        let e = between(
            column(0, ArrowType::Int32, true),
            lit(&Literal::Integer(5)),
            hi,
            &env,
        )
        .unwrap();
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [Some(false), Some(false)]
        );
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    /// `BETWEEN` is `v >= lo AND v <= hi` under Kleene logic: NULL in,
    /// NULL out, unless one side already decided `false`.
    #[test]
    fn between_follows_kleene_logic_for_nulls() {
        let env = env(true);
        let batch = rows_batch(1);
        for (v, lo, hi, want) in [
            (
                Literal::Long(2),
                Literal::Long(1),
                Literal::Null(None),
                None,
            ),
            (
                Literal::Long(2),
                Literal::Null(None),
                Literal::Long(3),
                None,
            ),
            (
                Literal::Null(None),
                Literal::Long(1),
                Literal::Long(3),
                None,
            ),
            (
                Literal::Long(5),
                Literal::Long(1),
                Literal::Long(3),
                Some(false),
            ),
            // `false AND NULL` is `false`: the lower bound already decided.
            (
                Literal::Long(0),
                Literal::Long(1),
                Literal::Null(None),
                Some(false),
            ),
            (
                Literal::Long(2),
                Literal::Long(1),
                Literal::Long(3),
                Some(true),
            ),
        ] {
            let e = between(lit(&v), lit(&lo), lit(&hi), &env).unwrap();
            assert_eq!(
                values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
                [want],
                "{v:?} BETWEEN {lo:?} AND {hi:?}"
            );
        }
    }

    /// The `BETWEEN` value is evaluated once per batch, for both bounds
    /// alike.
    #[test]
    fn between_evaluates_its_value_once() {
        let env = env(true);
        let batch = rows_batch(1);
        let n = Arc::new(AtomicUsize::new(0));
        let value: Arc<dyn PhysExpr> = Arc::new(Count {
            inner: lit(&Literal::Long(1)),
            n: Arc::clone(&n),
        });
        let e = between(value, lit(&Literal::Long(0)), lit(&Literal::Long(2)), &env).unwrap();
        assert_eq!(
            values(&e, &batch).as_boolean().iter().collect::<Vec<_>>(),
            [Some(true)]
        );
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }

    /// A `nullif` value of another Spark type than the result keeps no
    /// result cast: the mix fails at evaluation, as the analyzer never
    /// produces it.
    #[test]
    fn nullif_with_a_value_outside_the_result_type_fails_evaluation() {
        let env = env(true);
        let batch = rows_batch(1);
        let schema = Arc::new(Schema::empty());
        let f = resolved("nullif", DataType::String);
        let e = null_if(
            &f,
            vec![lit(&Literal::Integer(1)), lit(&Literal::Integer(1))],
            input(&env, &[], &schema),
        )
        .unwrap();
        assert_eq!(e.data_type(), &ArrowType::Utf8View);
        assert!(e.eval(&batch, 0).is_err());
    }
}
