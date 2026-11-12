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
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType as ArrowType;
use arrow_select::filter::filter_record_batch;
use arrow_select::interleave::interleave;
use arrow_select::zip::zip;
use krishiv_kernels::KernelCtx;
use krishiv_kernels::compare::CompareKind;
use krishiv_plan::{Builtin, CastMode, ResolvedFunction};

use super::cast::{CastParts, apply, canonical_carrier, parts};
use super::logic::{cmp, compare, is_null, long_bytes, view_parts};
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
    if items.iter().all(|e| e.is_constant())
        && let Some(views) = InViews::try_new(&v, &items)
    {
        return Ok(Arc::new(views));
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
pub(super) fn string_target<'a>(
    types: impl IntoIterator<Item = &'a ArrowType>,
) -> Option<ArrowType> {
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
pub(super) const IN_SET_THRESHOLD: usize = 10;

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
            ArrowType::Decimal64(..) => {
                let mut set = fast_set(values.len());
                set.extend(values.iter().map(|a| {
                    i128::from(
                        a.as_primitive::<arrow_array::types::Decimal64Type>()
                            .value(0),
                    )
                }));
                Keys::Decimals(set)
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
                if let Some(d) = a.as_primitive_opt::<arrow_array::types::Decimal64Type>() {
                    d.iter()
                        .for_each(|x| decide(x.map(|x| set.contains(&i128::from(x)))));
                } else {
                    let d = a.as_primitive::<arrow_array::types::Decimal128Type>();
                    d.iter().for_each(|x| decide(x.map(|x| set.contains(&x))));
                }
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

/// `v IN (constants)`, at most [`IN_SET_THRESHOLD`] of them, over a string or
/// binary view column: each row is compared in place against the list, with
/// no array per element. The same answers as [`InList`].
#[derive(Debug)]
struct InViews {
    v: Arc<dyn PhysExpr>,
    /// Values of 12 bytes or fewer, as their whole views.
    inline: Vec<u128>,
    /// Longer values: the view's low 64 bits (length and first four bytes) and the whole value.
    long: Vec<(u64, Vec<u8>)>,
    /// The list holds a NULL.
    has_null: bool,
}

/// A view's low 64 bits (length, first four bytes) and its length.
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "a view's 32/64-bit fields")]
pub(super) fn view_head(view: u128) -> (u64, u32) {
    (view as u64, view as u32)
}

impl InViews {
    /// `None` unless `v` is on a view type and every element evaluates to one
    /// value of that type here (as [`InSet::try_new`]).
    fn try_new(v: &Arc<dyn PhysExpr>, items: &[Arc<dyn PhysExpr>]) -> Option<Self> {
        let t = v.data_type();
        if !matches!(t, ArrowType::Utf8View | ArrowType::BinaryView) {
            return None;
        }
        let empty = RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty()));
        let mut out = Self {
            v: Arc::clone(v),
            inline: Vec::new(),
            long: Vec::new(),
            has_null: false,
        };
        for item in items {
            let a = Arc::clone(item.eval(&empty, 0).ok()?.array());
            if a.len() != 1 {
                return None;
            }
            if a.is_null(0) || *a.data_type() == ArrowType::Null {
                out.has_null = true;
                continue;
            }
            if a.data_type() != t {
                return None;
            }
            let (views, buffers) = view_parts(a.as_ref());
            let view = *views.first()?;
            if u32::try_from(view & u128::from(u32::MAX)).ok()? <= 12 {
                out.inline.push(view);
            } else {
                out.long
                    .push((view_head(view).0, long_bytes(view, buffers)?.to_vec()));
            }
        }
        // Sorted only so that dedup drops repeated literals (each costs a pass per batch).
        out.inline.sort_unstable();
        out.inline.dedup();
        Some(out)
    }
}

impl PhysExpr for InViews {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        self.v.nullable() || self.has_null
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.v.eval(batch, partition)?;
        let a = v.array();
        let (views, buffers) = view_parts(a.as_ref());
        let n = views.len();
        // One pass over the views per literal, 64 rows to a word: a plain compare the compiler
        // vectorises, with no per-row closure and no search of the list.
        let mut words: Vec<u64> = views
            .chunks(64)
            .map(|chunk| {
                let mut word = 0u64;
                for &k in &self.inline {
                    let mut hit = 0u64;
                    for (j, &row) in chunk.iter().enumerate() {
                        hit |= u64::from(row == k) << j;
                    }
                    word |= hit;
                }
                word
            })
            .collect();
        if !self.long.is_empty() {
            for (i, &view) in views.iter().enumerate() {
                // The length and first four bytes decide most rows without touching a buffer.
                let (head, len) = view_head(view);
                if len > 12
                    && self.long.iter().any(|(h, bytes)| {
                        *h == head
                            && long_bytes(view, buffers).is_some_and(|b| b == bytes.as_slice())
                    })
                    && let Some(word) = words.get_mut(i / 64)
                {
                    *word |= 1 << (i % 64);
                }
            }
        }
        let values = BooleanBuffer::new(arrow_buffer::Buffer::from_vec(words), 0, n);
        // A row that found nothing is NULL when the list holds a NULL.
        let nulls = if self.has_null {
            NullBuffer::union(
                a.logical_nulls().as_ref(),
                Some(&NullBuffer::new(values.clone())),
            )
        } else {
            a.logical_nulls()
        };
        let out: ArrayRef = Arc::new(BooleanArray::new(values, nulls));
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
mod tests;
