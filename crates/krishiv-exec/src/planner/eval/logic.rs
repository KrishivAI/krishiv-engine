// SPDX-License-Identifier: Apache-2.0
//! Boolean logic, `IS [NOT] NULL` and comparisons.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType as ArrowType;
use krishiv_kernels::compare::{
    self, BoolOperand, CompareKind, Equal, Greater, GreaterEq, Less, LessEq, Validity,
};
use krishiv_kernels::{KernelCtx, Operand};
use krishiv_plan::Builtin;

use super::{EvalEnv, PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::internal;

/// The values and validity of a boolean array (all valid without nulls).
fn parts(a: &BooleanArray) -> (BooleanBuffer, BooleanBuffer) {
    let valid = a
        .nulls()
        .map_or_else(|| BooleanBuffer::new_set(a.len()), |n| n.inner().clone());
    (a.values().clone(), valid)
}

/// Two boolean values of one batch as arrays of its row count.
fn both(l: &Value, r: &Value, rows: usize) -> Result<(BooleanArray, BooleanArray)> {
    let as_bool = |v: &Value| -> Result<BooleanArray> {
        v.clone()
            .into_array(rows)?
            .as_boolean_opt()
            .cloned()
            .ok_or_else(|| internal("a boolean operand is not boolean"))
    };
    Ok((as_bool(l)?, as_bool(r)?))
}

/// Kleene `AND` (`and`) or `OR`.
#[derive(Debug)]
struct AndOr {
    and: bool,
    l: Arc<dyn PhysExpr>,
    r: Arc<dyn PhysExpr>,
}

pub(super) fn and_or(and: bool, l: Arc<dyn PhysExpr>, r: Arc<dyn PhysExpr>) -> Arc<dyn PhysExpr> {
    Arc::new(AndOr { and, l, r })
}

impl PhysExpr for AndOr {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        self.l.nullable() || self.r.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let lv = self.l.eval(batch, partition)?;
        if lv.is_scalar() && batch.num_rows() > 0 {
            // A literal left side decides for every row or for none.
            let l = lv
                .array()
                .as_boolean_opt()
                .ok_or_else(|| internal("AND of a non-boolean"))?;
            let decides = l.is_valid(0) && l.value(0) != self.and;
            if decides {
                return Ok(lv);
            }
        }
        // The right side is evaluated only where the left one does not
        // decide (Spark's short circuit): `a <> 0 AND 10 / a > 1` never
        // divides by zero.
        let rows = batch.num_rows();
        let l_all = lv.clone().into_array(rows)?;
        let l = l_all
            .as_boolean_opt()
            .ok_or_else(|| internal("AND of a non-boolean"))?;
        let (lb, l_ok) = parts(l);
        let decided = if self.and { &l_ok & &!&lb } else { &l_ok & &lb };
        let open = !&decided;
        let rv = if open.count_set_bits() == rows {
            self.r.eval(batch, partition)?
        } else {
            Value::Array(super::eval_where(self.r.as_ref(), batch, &open, partition)?)
        };
        let scalar = lv.is_scalar() && rv.is_scalar();
        let rows = if scalar { 1 } else { rows };
        let (l, r) = both(&lv, &rv, rows)?;
        let ((lb, l_ok), (rb, rok)) = (parts(&l), parts(&r));
        // AND: false wins over NULL; OR: true wins over NULL.
        let (values, valid) = if self.and {
            let l_false = &l_ok & &!&lb;
            let r_false = &rok & &!&rb;
            let decided = &(&l_false | &r_false) | &(&l_ok & &rok);
            (&lb & &rb, decided)
        } else {
            let l_true = &l_ok & &lb;
            let r_true = &rok & &rb;
            let decided = &(&l_true | &r_true) | &(&l_ok & &rok);
            (&(&lb & &l_ok) | &(&rb & &rok), decided)
        };
        let out: ArrayRef = Arc::new(BooleanArray::new(values, Some(NullBuffer::new(valid))));
        Ok(if scalar {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}

/// `NOT`.
#[derive(Debug)]
struct Not(Arc<dyn PhysExpr>);

pub(super) fn not(e: Arc<dyn PhysExpr>) -> Arc<dyn PhysExpr> {
    Arc::new(Not(e))
}

impl PhysExpr for Not {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        self.0.nullable()
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.0.eval(batch, partition)?;
        let a = v
            .array()
            .as_boolean_opt()
            .ok_or_else(|| internal("NOT of a non-boolean"))?;
        let out: ArrayRef = Arc::new(BooleanArray::new(!a.values(), a.nulls().cloned()));
        Ok(match v {
            Value::Scalar(_) => Value::Scalar(out),
            Value::Array(_) => Value::Array(out),
        })
    }
}

/// `IS NULL` (`null`) or `IS NOT NULL`.
#[derive(Debug)]
struct IsNull {
    null: bool,
    e: Arc<dyn PhysExpr>,
}

pub(super) fn is_null(null: bool, e: Arc<dyn PhysExpr>) -> Arc<dyn PhysExpr> {
    Arc::new(IsNull { null, e })
}

impl PhysExpr for IsNull {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        false
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let v = self.e.eval(batch, partition)?;
        let a = v.array();
        let valid = match (a.data_type(), a.logical_nulls()) {
            (ArrowType::Null, _) => BooleanBuffer::new_unset(a.len()),
            (_, Some(n)) => n.inner().clone(),
            (_, None) => BooleanBuffer::new_set(a.len()),
        };
        let values = if self.null { !&valid } else { valid };
        let out: ArrayRef = Arc::new(BooleanArray::new(values, None));
        Ok(match v {
            Value::Scalar(_) => Value::Scalar(out),
            Value::Array(_) => Value::Array(out),
        })
    }
}

/// A comparison of two values of one type.
#[derive(Debug)]
struct Compare {
    /// `None`: `<=>`.
    kind: Option<CompareKind>,
    l: Arc<dyn PhysExpr>,
    r: Arc<dyn PhysExpr>,
    kernels: KernelCtx,
}

pub(super) fn compare(
    op: Builtin,
    l: Arc<dyn PhysExpr>,
    r: Arc<dyn PhysExpr>,
    env: &EvalEnv,
) -> Result<Arc<dyn PhysExpr>> {
    let kind = match op {
        Builtin::Eq => Some(CompareKind::Equal),
        Builtin::Lt => Some(CompareKind::Less),
        Builtin::LtEq => Some(CompareKind::LessEq),
        Builtin::Gt => Some(CompareKind::Greater),
        Builtin::GtEq => Some(CompareKind::GreaterEq),
        _ => None,
    };
    // Operands of one Spark `STRING`/`BINARY` in different Arrow encodings
    // (a `Utf8` column beside a `Utf8View` literal) meet in the canonical
    // carrier; every other mismatch still fails below.
    let (l, r) = super::cast::unify(l, r, env)?;
    let (lt, rt) = (l.data_type(), r.data_type());
    if matches!((lt, rt), (ArrowType::Struct(_), ArrowType::Struct(_))) {
        if !struct_compatible(lt, rt) {
            return Err(internal(&format!("comparison of {lt} with {rt}")));
        }
        if !krishiv_kernels::r#struct::comparable(lt) {
            return Err(crate::planner::build::unsupported(&format!(
                "comparison of {lt} values"
            )));
        }
    }
    let comparable = lt == rt
        || matches!((lt, rt), (ArrowType::Null, _) | (_, ArrowType::Null))
        || matches!(
            (lt, rt),
            (ArrowType::Timestamp(lu, _), ArrowType::Timestamp(ru, _)) if lu == ru
        )
        || matches!((lt, rt), (ArrowType::Struct(_), ArrowType::Struct(_)));
    if !comparable {
        return Err(internal(&format!("comparison of {lt} with {rt}")));
    }
    Ok(Arc::new(Compare {
        kind,
        l,
        r,
        kernels: env.kernels,
    }))
}

impl PhysExpr for Compare {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        self.kind.is_some() && (maybe_null(&self.l) || maybe_null(&self.r))
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let (mut l, r) = (
            self.l.eval(batch, partition)?,
            self.r.eval(batch, partition)?,
        );
        let scalar = l.is_scalar() && r.is_scalar();
        if scalar {
            // Kernels take at most one literal side.
            l = Value::Array(Arc::clone(l.array()));
        }
        let eq = cmp(
            self.kind.unwrap_or(CompareKind::Equal),
            &l,
            &r,
            &self.kernels,
        )?;
        let out = if self.kind.is_some() {
            eq
        } else {
            compare::null_safe(&eq, validity(&l), validity(&r))?
        };
        let out: ArrayRef = Arc::new(out);
        Ok(if scalar {
            Value::Scalar(out)
        } else {
            Value::Array(out)
        })
    }
}

/// The validity of a `<=>` operand.
fn validity(v: &Value) -> Validity<'_> {
    match v {
        // An untyped NULL is a `NullArray`, whose nulls are logical only.
        Value::Scalar(s) => Validity::Scalar {
            null: s.logical_null_count() > 0,
        },
        Value::Array(a) if a.data_type() == &ArrowType::Null => Validity::Scalar { null: true },
        Value::Array(a) => Validity::Array(a.nulls()),
    }
}

/// A boolean operand.
fn bool_operand(v: &Value) -> BoolOperand<'_> {
    match v {
        Value::Array(a) => BoolOperand::Array(a.as_boolean()),
        Value::Scalar(s) => {
            let b = s.as_boolean();
            BoolOperand::Scalar(b.is_valid(0).then(|| b.value(0)))
        }
    }
}

/// A primitive operand.
pub(super) fn prim<T: arrow_array::ArrowPrimitiveType>(v: &Value) -> Operand<'_, T> {
    match v {
        Value::Array(a) => Operand::Array(a.as_primitive::<T>()),
        Value::Scalar(s) => {
            let p = s.as_primitive::<T>();
            Operand::Scalar(p.is_valid(0).then(|| p.value(0)))
        }
    }
}

/// Whether the value of `e` may be null.
fn maybe_null(e: &Arc<dyn PhysExpr>) -> bool {
    e.nullable()
}

/// `l kind r` (NULL where either is NULL).
pub(crate) fn cmp(
    kind: CompareKind,
    l: &Value,
    r: &Value,
    ctx: &KernelCtx,
) -> Result<BooleanArray> {
    // The primitive kernels need an array operand: of two scalars, one
    // goes in as its own one-row array (no copy), so the answer is one row.
    if let (Value::Scalar(one), Value::Scalar(_)) = (l, r) {
        return cmp(kind, &Value::Array(Arc::clone(one)), r, ctx);
    }
    macro_rules! prim {
        ($t:ty) => {{
            let (a, b) = (prim::<$t>(l), prim::<$t>(r));
            match kind {
                CompareKind::Equal => compare::compare::<$t, Equal>(a, b, ctx),
                CompareKind::NotEqual => compare::compare::<$t, compare::NotEqual>(a, b, ctx),
                CompareKind::Less => compare::compare::<$t, Less>(a, b, ctx),
                CompareKind::LessEq => compare::compare::<$t, LessEq>(a, b, ctx),
                CompareKind::Greater => compare::compare::<$t, Greater>(a, b, ctx),
                CompareKind::GreaterEq => compare::compare::<$t, GreaterEq>(a, b, ctx),
            }?
        }};
    }
    let rows = match (l, r) {
        (Value::Array(a), _) | (_, Value::Array(a)) => a.len(),
        _ => 1,
    };
    let t = if l.array().data_type() == &ArrowType::Null {
        r.array().data_type()
    } else {
        l.array().data_type()
    };
    if l.array().data_type() == &ArrowType::Null || r.array().data_type() == &ArrowType::Null {
        return Ok(BooleanArray::new_null(rows));
    }
    Ok(match t {
        ArrowType::Int8 => prim!(Int8Type),
        ArrowType::Int16 => prim!(Int16Type),
        ArrowType::Int32 => prim!(Int32Type),
        ArrowType::Int64 => prim!(Int64Type),
        ArrowType::Float32 => prim!(Float32Type),
        ArrowType::Float64 => prim!(Float64Type),
        ArrowType::Decimal64(..) => prim!(Decimal64Type),
        ArrowType::Decimal128(..) => prim!(Decimal128Type),
        ArrowType::Date32 => prim!(Date32Type),
        ArrowType::Timestamp(..) => prim!(TimestampMicrosecondType),
        ArrowType::Interval(arrow_schema::IntervalUnit::YearMonth) => {
            prim!(arrow_array::types::IntervalYearMonthType)
        }
        ArrowType::Duration(arrow_schema::TimeUnit::Microsecond) => {
            prim!(arrow_array::types::DurationMicrosecondType)
        }
        ArrowType::Boolean => {
            let (a, b) = (bool_operand(l), bool_operand(r));
            match kind {
                CompareKind::Equal => compare::compare_bool::<Equal>(a, b),
                CompareKind::NotEqual => compare::compare_bool::<compare::NotEqual>(a, b),
                CompareKind::Less => compare::compare_bool::<Less>(a, b),
                CompareKind::LessEq => compare::compare_bool::<LessEq>(a, b),
                CompareKind::Greater => compare::compare_bool::<Greater>(a, b),
                CompareKind::GreaterEq => compare::compare_bool::<GreaterEq>(a, b),
            }?
        }
        ArrowType::Utf8View | ArrowType::BinaryView
            if matches!(kind, CompareKind::Equal | CompareKind::NotEqual)
                && l.array().data_type() == r.array().data_type() =>
        {
            view_eq(kind == CompareKind::Equal, l, r, rows)
        }
        ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8 => {
            bytes(kind, l, r, rows, |a, i| str_value(a, i).as_bytes())?
        }
        ArrowType::BinaryView | ArrowType::Binary | ArrowType::LargeBinary => {
            bytes(kind, l, r, rows, bin_value)?
        }
        ArrowType::Struct(fields) => struct_cmp(kind, l, r, rows, fields.len(), ctx)?,
        other => {
            return Err(crate::planner::build::unsupported(&format!(
                "comparison of {other} values"
            )));
        }
    })
}

/// Whether struct Arrow types compare field-wise: the same fields of
/// recursively compatible types (names and nullability do not matter, as
/// the analyzer's shape check already allowed them).
fn struct_compatible(l: &ArrowType, r: &ArrowType) -> bool {
    match (l, r) {
        (ArrowType::Struct(l), ArrowType::Struct(r)) => {
            l.len() == r.len()
                && l.iter()
                    .zip(r.iter())
                    .all(|(a, b)| struct_compatible(a.data_type(), b.data_type()))
        }
        (a, b) => a == b,
    }
}

/// `l kind r` for structs of `fields` fields: every field through the
/// existing comparator, combined without boxing rows. Spark orders nulls as
/// the smallest value and never returns NULL for a null field: two null
/// fields are equal, a null against a value is unequal (and smaller for
/// ordering). The result is NULL only where a whole struct operand is NULL
/// (the `<=>` caller turns those into `true`/`false` through `null_safe`).
fn struct_cmp(
    kind: CompareKind,
    l: &Value,
    r: &Value,
    rows: usize,
    fields: usize,
    ctx: &KernelCtx,
) -> Result<BooleanArray> {
    let array = |v: &Value| v.clone().into_array(rows);
    let (l, r) = (array(l)?, array(r)?);
    let (ls, rs) = (l.as_struct(), r.as_struct());
    if ls.len() != rows
        || rs.len() != rows
        || ls.num_columns() != fields
        || rs.num_columns() != fields
    {
        return Err(internal("struct comparison operands differ in shape"));
    }
    let field = |s: &arrow_array::StructArray, i: usize| {
        krishiv_kernels::r#struct::get_field(s, i).map_err(|e| internal(&e.to_string()))
    };
    // Logical nulls: a `NullArray` field (an untyped NULL) has no null
    // buffer but every row is NULL.
    let valid_of = |a: &arrow_array::ArrayRef| match a.logical_nulls() {
        Some(n) => n.into_inner(),
        None => BooleanBuffer::new_set(a.len()),
    };
    let ls_valid = valid_of(&l);
    let rs_valid = valid_of(&r);
    let struct_null: BooleanBuffer = &!&ls_valid | &!&rs_valid;
    let both_structs: BooleanBuffer = &ls_valid & &rs_valid;
    if kind == CompareKind::Equal || kind == CompareKind::NotEqual {
        let mut any_unequal = BooleanBuffer::new_unset(rows);
        for i in 0..fields {
            let (fl, fr) = (field(ls, i)?, field(rs, i)?);
            let (fl_valid, fr_valid) = (valid_of(&fl), valid_of(&fr));
            let eq = cmp(
                CompareKind::Equal,
                &Value::Array(Arc::clone(&fl)),
                &Value::Array(Arc::clone(&fr)),
                ctx,
            )?;
            let eq_values = eq.values().clone();
            let both_null: BooleanBuffer = &!&fl_valid & &!&fr_valid;
            let both_equal: BooleanBuffer = &(&fl_valid & &fr_valid) & &eq_values;
            let field_equal: BooleanBuffer = &both_null | &both_equal;
            let unequal: BooleanBuffer = &!&field_equal & &both_structs;
            any_unequal |= &unequal;
        }
        let values = if kind == CompareKind::Equal {
            !&any_unequal
        } else {
            any_unequal.clone()
        };
        return Ok(BooleanArray::new(values, {
            (struct_null.count_set_bits() > 0).then(|| NullBuffer::new(!&struct_null))
        }));
    }
    let mut decided = BooleanBuffer::new_unset(rows);
    let mut values = BooleanBuffer::new_unset(rows);
    let less_kind = matches!(kind, CompareKind::Less | CompareKind::LessEq);
    for i in 0..fields {
        let (fl, fr) = (field(ls, i)?, field(rs, i)?);
        let (fl_valid, fr_valid) = (valid_of(&fl), valid_of(&fr));
        let lt = cmp(
            CompareKind::Less,
            &Value::Array(Arc::clone(&fl)),
            &Value::Array(Arc::clone(&fr)),
            ctx,
        )?;
        let eq = cmp(
            CompareKind::Equal,
            &Value::Array(Arc::clone(&fl)),
            &Value::Array(Arc::clone(&fr)),
            ctx,
        )?;
        let (lt_values, lt_valid) = parts(&lt);
        let (eq_values, _) = parts(&eq);
        let not_decided: BooleanBuffer = &!&decided & &both_structs;
        // Nulls order smallest: null < value, null == null continues.
        let l_null_only: BooleanBuffer = &(&!&fl_valid & &fr_valid) & &not_decided;
        let r_null_only: BooleanBuffer = &(&fl_valid & &!&fr_valid) & &not_decided;
        let both_valid: BooleanBuffer = &(&fl_valid & &fr_valid) & &not_decided;
        let less_both: BooleanBuffer = &(&lt_values & &lt_valid) & &both_valid;
        let equal_both: BooleanBuffer = &(&eq_values & &both_valid) & &both_valid;
        let greater_both: BooleanBuffer = &(&both_valid & &!&less_both) & &!&equal_both;
        let less: BooleanBuffer = &l_null_only | &less_both;
        let greater: BooleanBuffer = &r_null_only | &greater_both;
        let (hit, miss) = if less_kind {
            (less, greater)
        } else {
            (greater, less)
        };
        values |= &hit;
        decided |= &hit;
        decided |= &miss;
        if decided.count_set_bits() == rows {
            break;
        }
    }
    // Equal structs (every field both null or equal) hold `<=`/`>=`.
    if matches!(kind, CompareKind::LessEq | CompareKind::GreaterEq) {
        let undecided: BooleanBuffer = &both_structs & &!&decided;
        values |= &undecided;
        decided |= &undecided;
    }
    Ok(BooleanArray::new(values, {
        (struct_null.count_set_bits() > 0).then(|| NullBuffer::new(!&struct_null))
    }))
}

/// The string at row `i` of a `Utf8`/`LargeUtf8`/`Utf8View` array.
pub(super) fn str_value(a: &dyn Array, i: usize) -> &str {
    match a.data_type() {
        ArrowType::Utf8View => a.as_string_view().value(i),
        ArrowType::LargeUtf8 => a.as_string::<i64>().value(i),
        _ => a.as_string::<i32>().value(i),
    }
}

/// The bytes at row `i` of a binary array.
pub(super) fn bin_value(a: &dyn Array, i: usize) -> &[u8] {
    match a.data_type() {
        ArrowType::BinaryView => a.as_binary_view().value(i),
        ArrowType::LargeBinary => a.as_binary::<i64>().value(i),
        _ => a.as_binary::<i32>().value(i),
    }
}

/// The views and data buffers of a string or binary view array.
pub(super) fn view_parts(a: &dyn Array) -> (&[u128], &[arrow_buffer::Buffer]) {
    match a.data_type() {
        ArrowType::Utf8View => {
            let v = a.as_string_view();
            (v.views(), v.data_buffers())
        }
        _ => {
            let v = a.as_binary_view();
            (v.views(), v.data_buffers())
        }
    }
}

/// Whether two views hold equal bytes. Arrow's view layout: the low 32
/// bits are the length; a value of 12 bytes or fewer sits inline,
/// zero-padded, so equal values have equal views; a longer one keeps its
/// first 4 bytes in the view (bits 32..64) and the rest in a data buffer
/// (buffer index in bits 64..96, offset in bits 96..128).
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "a view's 32-bit fields")]
fn views_equal(a: u128, ab: &[arrow_buffer::Buffer], b: u128, bb: &[arrow_buffer::Buffer]) -> bool {
    let len = a as u32;
    if len != b as u32 {
        return false;
    }
    if len <= 12 {
        return a == b;
    }
    // Length and prefix first: most unequal values differ there.
    if a as u64 != b as u64 {
        return false;
    }
    // A view that points outside its buffers equals nothing.
    matches!((long_bytes(a, ab), long_bytes(b, bb)), (Some(x), Some(y)) if x == y)
}

/// The bytes of a view of a value longer than 12 bytes, in its data buffer.
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "a view's 32-bit fields")]
pub(super) fn long_bytes(v: u128, bufs: &[arrow_buffer::Buffer]) -> Option<&[u8]> {
    let (len, buffer, offset) = (
        v as u32 as usize,
        (v >> 64) as u32 as usize,
        (v >> 96) as u32 as usize,
    );
    bufs.get(buffer)?.get(offset..offset.checked_add(len)?)
}

/// `=` (`equal`) or `<>` of two string or binary view operands of one
/// type, view against view: no per-row dispatch, and most rows decided
/// from the 16-byte views alone (the arrow-rs view comparison's method).
fn view_eq(equal: bool, l: &Value, r: &Value, rows: usize) -> BooleanArray {
    let (la, ra) = (l.array(), r.array());
    let ((lv, lb), (rv, rb)) = (view_parts(la.as_ref()), view_parts(ra.as_ref()));
    let (ls, rs) = (l.is_scalar(), r.is_scalar());
    let values = match (ls, rs, lv.first(), rv.first()) {
        (false, true, _, Some(&b)) => view_eq_literal(equal, lv, lb, b, rb, rows),
        (true, false, Some(&a), _) => view_eq_literal(equal, rv, rb, a, lb, rows),
        _ => {
            let at = |scalar: bool, i: usize| if scalar { 0 } else { i };
            BooleanBuffer::collect_bool(rows, |i| {
                let same = match (lv.get(at(ls, i)), rv.get(at(rs, i))) {
                    (Some(&a), Some(&b)) => views_equal(a, lb, b, rb),
                    _ => false,
                };
                same == equal
            })
        }
    };
    let null_of = |a: &ArrayRef, scalar: bool| -> Option<NullBuffer> {
        if scalar {
            // A NULL scalar makes every row NULL.
            a.is_null(0).then(|| NullBuffer::new_null(rows))
        } else {
            a.logical_nulls()
        }
    };
    let nulls = NullBuffer::union(null_of(la, ls).as_ref(), null_of(ra, rs).as_ref());
    BooleanArray::new(values, nulls)
}

/// [`view_eq`] of a column and one value: a value of 12 bytes or fewer is
/// decided by the whole view alone.
#[expect(clippy::cast_possible_truncation, reason = "a view's 32-bit fields")]
fn view_eq_literal(
    equal: bool,
    views: &[u128],
    bufs: &[arrow_buffer::Buffer],
    lit: u128,
    lit_bufs: &[arrow_buffer::Buffer],
    rows: usize,
) -> BooleanBuffer {
    let views = views.get(..rows).unwrap_or(views);
    if lit as u32 <= 12 {
        return BooleanBuffer::collect_bool(views.len(), |i| {
            views.get(i).is_some_and(|&v| v == lit) == equal
        });
    }
    BooleanBuffer::collect_bool(views.len(), |i| {
        views
            .get(i)
            .is_some_and(|&v| views_equal(v, bufs, lit, lit_bufs))
            == equal
    })
}

/// Byte-wise comparison (Spark's `UTF8_BINARY` order for strings).
fn bytes(
    kind: CompareKind,
    l: &Value,
    r: &Value,
    rows: usize,
    at: for<'a> fn(&'a dyn Array, usize) -> &'a [u8],
) -> Result<BooleanArray> {
    let side = |v: &Value| -> (ArrayRef, bool) { (Arc::clone(v.array()), v.is_scalar()) };
    let ((la, ls), (ra, rs)) = (side(l), side(r));
    let row = |scalar: bool, i: usize| if scalar { 0 } else { i };
    let valid = |a: &ArrayRef, scalar: bool, i: usize| a.is_valid(row(scalar, i));
    let mut values = Vec::with_capacity(rows);
    let mut nulls = Vec::with_capacity(rows);
    for i in 0..rows {
        let ok = valid(&la, ls, i) && valid(&ra, rs, i);
        nulls.push(ok);
        values.push(ok && kind.holds(at(la.as_ref(), row(ls, i)).cmp(at(ra.as_ref(), row(rs, i)))));
    }
    Ok(BooleanArray::new(
        BooleanBuffer::from(values),
        Some(NullBuffer::from(nulls)),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{
        Array, ArrayRef, BooleanArray, Int32Array, NullArray, RecordBatch, StringArray,
        StringViewArray, StructArray,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType as ArrowType, Field, Fields, Schema};
    use krishiv_common::config::Config;

    use super::*;

    /// An input column.
    #[derive(Debug)]
    struct Col {
        index: usize,
        data_type: ArrowType,
    }

    impl PhysExpr for Col {
        fn data_type(&self) -> &ArrowType {
            &self.data_type
        }

        fn nullable(&self) -> bool {
            true
        }

        fn eval(&self, batch: &RecordBatch, _partition: u32) -> Result<Value> {
            Ok(Value::Array(Arc::clone(batch.column(self.index))))
        }
    }

    fn env() -> EvalEnv {
        EvalEnv::new(&Config::new(), Arc::from("test"))
    }

    /// Operands of one Spark `STRING` in different Arrow encodings compare
    /// by value; operands of different Spark types still fail.
    #[test]
    fn comparisons_unify_string_encodings() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("u", ArrowType::Utf8, true),
            Field::new("v", ArrowType::Utf8View, true),
            Field::new("i", ArrowType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec![Some("b"), None])) as ArrayRef,
                Arc::new(StringViewArray::from(vec![Some("b"), Some("c")])),
                Arc::new(Int32Array::from(vec![Some(1), Some(2)])),
            ],
        )
        .unwrap();
        let env = env();
        let utf8: Arc<dyn PhysExpr> = Arc::new(Col {
            index: 0,
            data_type: ArrowType::Utf8,
        });
        let view: Arc<dyn PhysExpr> = Arc::new(Col {
            index: 1,
            data_type: ArrowType::Utf8View,
        });
        for (op, want, row1) in [
            (Builtin::Eq, true, None),
            (Builtin::EqNullSafe, true, Some(false)),
            (Builtin::Lt, false, None),
            (Builtin::LtEq, true, None),
            (Builtin::Gt, false, None),
            (Builtin::GtEq, true, None),
        ] {
            let e = compare(op, Arc::clone(&utf8), Arc::clone(&view), &env).unwrap();
            let got = e.eval(&batch, 0).unwrap().into_array(2).unwrap();
            let got = got.as_boolean();
            assert_eq!(got.len(), 2, "{op:?}");
            assert_eq!(got.value(0), want, "{op:?}");
            assert_eq!(got.is_valid(1).then(|| got.value(1)), row1, "{op:?}");
        }
        // `NULL = 'c'` is NULL, `NULL <=> 'c'` is false.
        let e = compare(Builtin::Eq, Arc::clone(&utf8), Arc::clone(&view), &env).unwrap();
        assert!(
            e.eval(&batch, 0)
                .unwrap()
                .into_array(2)
                .unwrap()
                .as_boolean()
                .is_null(1)
        );
        // A `Utf8` literal (always a view) meets a `Utf8` column.
        let lit: Arc<dyn PhysExpr> = Arc::new(
            super::super::Lit::new(&krishiv_plan::Literal::String("b".to_owned())).unwrap(),
        );
        let e = compare(Builtin::Eq, Arc::clone(&utf8), lit, &env).unwrap();
        let got = e.eval(&batch, 0).unwrap().into_array(2).unwrap();
        assert_eq!(
            got.as_boolean().iter().collect::<Vec<_>>(),
            [Some(true), None]
        );
        // Different Spark types still fail as before.
        let int: Arc<dyn PhysExpr> = Arc::new(Col {
            index: 2,
            data_type: ArrowType::Int32,
        });
        let err = compare(Builtin::Eq, Arc::clone(&int), Arc::clone(&utf8), &env).unwrap_err();
        assert!(
            err.to_string().contains("comparison of Int32 with Utf8"),
            "{err}"
        );
    }

    fn is_null_scalar(v: &Value) -> bool {
        matches!(validity(v), Validity::Scalar { null: true })
    }

    #[test]
    fn an_untyped_null_operand_of_null_safe_equality_is_null() {
        assert!(is_null_scalar(&Value::Scalar(Arc::new(NullArray::new(1)))));
        assert!(is_null_scalar(&Value::Array(Arc::new(NullArray::new(3)))));
        let typed: ArrayRef = Arc::new(Int32Array::from(vec![None::<i32>]));
        assert!(is_null_scalar(&Value::Scalar(typed)));
        let one: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        assert!(!is_null_scalar(&Value::Scalar(one)));
    }

    fn struct_fields() -> Fields {
        vec![
            Field::new("a", ArrowType::Int32, true),
            Field::new("b", ArrowType::Int32, true),
        ]
        .into()
    }

    fn struct_array(a: Vec<Option<i32>>, b: Vec<Option<i32>>, valid: Vec<bool>) -> ArrayRef {
        let fields = struct_fields();
        let cols: Vec<ArrayRef> =
            vec![Arc::new(Int32Array::from(a)), Arc::new(Int32Array::from(b))];
        let nulls = if valid.iter().all(|v| *v) {
            None
        } else {
            Some(NullBuffer::from(valid))
        };
        Arc::new(StructArray::new(fields, cols, nulls))
    }

    #[test]
    fn struct_comparison_treats_null_fields_as_smallest_without_null() {
        let ctx = KernelCtx::new();
        // (1,null)|(1,2), (null,1)|(1,1), (1,2)|(1,null), (1,null)|(1,null),
        // (1,null)|(1,2), (1,null)|(2,null), null|(1,1), null|null.
        let l = struct_array(
            vec![
                Some(1),
                None,
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
            ],
            vec![None, Some(1), Some(2), None, None, None, Some(1), Some(1)],
            vec![true, true, true, true, true, true, false, false],
        );
        let r = struct_array(
            vec![
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(2),
                Some(1),
                Some(1),
            ],
            vec![
                Some(2),
                Some(1),
                None,
                None,
                Some(2),
                None,
                Some(1),
                Some(1),
            ],
            vec![true, true, true, true, true, true, true, false],
        );
        let lv = Value::Array(l);
        let rv = Value::Array(r);
        let eq = cmp(CompareKind::Equal, &lv, &rv, &ctx).expect("struct eq");
        assert_eq!(
            eq,
            BooleanArray::from(vec![
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                None,
                None,
            ])
        );
        let ne = cmp(CompareKind::NotEqual, &lv, &rv, &ctx).expect("struct ne");
        assert_eq!(
            ne,
            BooleanArray::from(vec![
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                Some(true),
                Some(true),
                None,
                None,
            ])
        );
        let lt = cmp(CompareKind::Less, &lv, &rv, &ctx).expect("struct lt");
        assert_eq!(
            lt,
            BooleanArray::from(vec![
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                Some(true),
                Some(true),
                None,
                None,
            ])
        );
        let gt = cmp(CompareKind::Greater, &lv, &rv, &ctx).expect("struct gt");
        assert_eq!(
            gt,
            BooleanArray::from(vec![
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                None,
                None,
            ])
        );
        let le = cmp(CompareKind::LessEq, &lv, &rv, &ctx).expect("struct le");
        assert_eq!(
            le,
            BooleanArray::from(vec![
                Some(true),
                Some(true),
                Some(false),
                Some(true),
                Some(true),
                Some(true),
                None,
                None,
            ])
        );
        let ge = cmp(CompareKind::GreaterEq, &lv, &rv, &ctx).expect("struct ge");
        assert_eq!(
            ge,
            BooleanArray::from(vec![
                Some(false),
                Some(false),
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                None,
                None,
            ])
        );
        // `<=>` from the equality above: never NULL, true only where both
        // structs are NULL or equal as values.
        let safe = compare::null_safe(&eq, validity(&lv), validity(&rv)).expect("null safe");
        assert_eq!(
            safe,
            BooleanArray::from(vec![false, false, false, true, false, false, false, true])
        );
    }

    /// `named_struct('a', NULL)` against itself: an untyped NULL field (a
    /// `NullArray`, with no null buffer) is NULL on every row, so the two
    /// fields compare equal, and a later field decides the order.
    #[test]
    fn untyped_null_fields_compare_as_equal_nulls() {
        let ctx = KernelCtx::new();
        let fields: Fields = vec![
            Field::new("a", ArrowType::Null, true),
            Field::new("b", ArrowType::Int32, true),
        ]
        .into();
        let make = |b: Vec<i32>| -> Value {
            let cols: Vec<ArrayRef> = vec![
                Arc::new(NullArray::new(b.len())),
                Arc::new(Int32Array::from(b)),
            ];
            Value::Array(Arc::new(StructArray::new(fields.clone(), cols, None)))
        };
        let (l, r) = (make(vec![1, 1]), make(vec![1, 2]));
        let at = |kind| cmp(kind, &l, &r, &ctx).expect("struct cmp");
        assert_eq!(
            at(CompareKind::Equal),
            BooleanArray::from(vec![Some(true), Some(false)])
        );
        assert_eq!(
            at(CompareKind::Less),
            BooleanArray::from(vec![Some(false), Some(true)])
        );
        assert_eq!(
            at(CompareKind::Greater),
            BooleanArray::from(vec![Some(false), Some(false)])
        );
        assert_eq!(
            at(CompareKind::LessEq),
            BooleanArray::from(vec![Some(true), Some(true)])
        );
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig { cases: 256, ..Default::default() })]
        /// The view comparison answers `=` and `<>` exactly as the row-wise
        /// byte comparison does: short (inline) and long values, shared
        /// prefixes, NULLs, and a scalar on either side.
        #[test]
        fn view_equality_matches_the_byte_comparison(
            left in proptest::collection::vec(proptest::option::of(value()), 0..40),
            right in proptest::collection::vec(proptest::option::of(value()), 0..40),
            scalar in 0u8..3,
            binary in proptest::bool::ANY,
        ) {
            let n = left.len().min(right.len());
            let make = |v: &[Option<String>]| -> ArrayRef {
                if binary {
                    Arc::new(arrow_array::BinaryViewArray::from_iter(v.iter().map(|s| s.as_deref().map(str::as_bytes))))
                } else {
                    Arc::new(StringViewArray::from_iter(v.iter().map(|s| s.as_deref())))
                }
            };
            let (l, r) = match scalar {
                0 => (Value::Array(make(&left[..n])), Value::Array(make(&right[..n]))),
                1 if !right.is_empty() => (Value::Array(make(&left[..n])), Value::Scalar(make(&right[..1]))),
                _ if !left.is_empty() => (Value::Scalar(make(&left[..1])), Value::Array(make(&right[..n]))),
                _ => (Value::Array(make(&left[..n])), Value::Array(make(&right[..n]))),
            };
            let rows = n;
            let at: for<'a> fn(&'a dyn Array, usize) -> &'a [u8] = if binary { bin_value } else { |a, i| str_value(a, i).as_bytes() };
            for (equal, kind) in [(true, CompareKind::Equal), (false, CompareKind::NotEqual)] {
                let got = view_eq(equal, &l, &r, rows);
                let want = bytes(kind, &l, &r, rows, at).unwrap();
                proptest::prop_assert_eq!(got.len(), want.len());
                for i in 0..rows {
                    proptest::prop_assert_eq!(got.is_valid(i), want.is_valid(i), "row {}", i);
                    if want.is_valid(i) {
                        proptest::prop_assert_eq!(got.value(i), want.value(i), "row {}", i);
                    }
                }
            }
        }
    }

    /// Sliced view arrays (an offset into shared views and buffers) compare
    /// as their rows, against a column and against a scalar.
    #[test]
    fn sliced_views_compare_their_rows() {
        let long = "a value longer than twelve bytes";
        let all = StringViewArray::from(vec![
            Some("x"),
            Some(long),
            None,
            Some("short"),
            Some(long),
            Some("a value longer than twelve bytes, but not equal"),
        ]);
        let left: ArrayRef = Arc::new(all.slice(1, 5));
        let right: ArrayRef = Arc::new(all.slice(0, 5));
        let lit: ArrayRef = Arc::new(StringViewArray::from(vec![long]));
        let at: for<'a> fn(&'a dyn Array, usize) -> &'a [u8] = |a, i| str_value(a, i).as_bytes();
        for (l, r) in [
            (Value::Array(Arc::clone(&left)), Value::Array(right)),
            (Value::Array(left), Value::Scalar(lit)),
        ] {
            for (equal, kind) in [(true, CompareKind::Equal), (false, CompareKind::NotEqual)] {
                let got = view_eq(equal, &l, &r, 5);
                let want = bytes(kind, &l, &r, 5, at).unwrap();
                assert_eq!(got, want, "{kind:?}");
            }
        }
    }

    /// Values around the 12-byte inline limit, sharing prefixes.
    fn value() -> impl proptest::strategy::Strategy<Value = String> {
        use proptest::strategy::Strategy as _;
        (
            proptest::sample::select(vec![
                "",
                "ab",
                "abcd",
                "abcdefghijkl",
                "abcdefghijklm",
                "abcdefghijklmnopqrstuv",
            ]),
            0u8..3,
        )
            .prop_map(|(base, tail)| format!("{base}{}", "x".repeat(usize::from(tail))))
    }
}
