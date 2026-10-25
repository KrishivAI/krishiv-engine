// SPDX-License-Identifier: Apache-2.0
//! Boolean logic, `IS [NOT] NULL` and comparisons.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType,
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

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let lv = self.l.eval(batch)?;
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
            self.r.eval(batch)?
        } else {
            Value::Array(super::eval_where(self.r.as_ref(), batch, &open)?)
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

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let v = self.0.eval(batch)?;
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

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let v = self.e.eval(batch)?;
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

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let (mut l, r) = (self.l.eval(batch)?, self.r.eval(batch)?);
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

/// Whether the value of `e` may be null: the expression may be, or a struct
/// has a field that may be (comparisons read fields, and a null field
/// decides them).
fn maybe_null(e: &Arc<dyn PhysExpr>) -> bool {
    e.nullable() || fields_nullable(e.data_type())
}

/// Whether the struct Arrow type `t` has a field that may be null, at any
/// depth.
fn fields_nullable(t: &ArrowType) -> bool {
    match t {
        ArrowType::Struct(fields) => fields
            .iter()
            .any(|f| f.is_nullable() || fields_nullable(f.data_type())),
        _ => false,
    }
}

/// `l kind r` (NULL where either is NULL).
pub(super) fn cmp(
    kind: CompareKind,
    l: &Value,
    r: &Value,
    ctx: &KernelCtx,
) -> Result<BooleanArray> {
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
        ArrowType::Decimal128(..) => prim!(Decimal128Type),
        ArrowType::Date32 => prim!(Date32Type),
        ArrowType::Timestamp(..) => prim!(TimestampMicrosecondType),
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
/// existing comparator, combined without boxing rows. Equality is the Kleene
/// conjunction of the field equalities (`false` where any field is unequal,
/// `null` where none is but some is null); ordering is lexicographic over
/// the first field that decides (a null field comparison decides nothing,
/// so those rows stay null).
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
    if kind == CompareKind::Equal || kind == CompareKind::NotEqual {
        let mut any_false = BooleanBuffer::new_unset(rows);
        let mut any_null = BooleanBuffer::new_unset(rows);
        for i in 0..fields {
            let eq = cmp(
                CompareKind::Equal,
                &Value::Array(field(ls, i)?),
                &Value::Array(field(rs, i)?),
                ctx,
            )?;
            let (values, valid) = parts(&eq);
            let diff: BooleanBuffer = &valid & &!&values;
            any_false |= &diff;
            let null: BooleanBuffer = !&valid;
            any_null |= &null;
        }
        let mut values: BooleanBuffer = !&any_false;
        if kind == CompareKind::NotEqual {
            values = !&values;
        }
        let none_false: BooleanBuffer = !&any_false;
        let is_null: BooleanBuffer = &any_null & &none_false;
        return Ok(BooleanArray::new(values, {
            (is_null.count_set_bits() > 0).then(|| NullBuffer::new(!&is_null))
        }));
    }
    let mut decided = BooleanBuffer::new_unset(rows);
    let mut values = BooleanBuffer::new_unset(rows);
    let mut nulls = BooleanBuffer::new_unset(rows);
    for i in 0..fields {
        let fl = Value::Array(field(ls, i)?);
        let fr = Value::Array(field(rs, i)?);
        let c = cmp(kind, &fl, &fr, ctx)?;
        let eq = cmp(CompareKind::Equal, &fl, &fr, ctx)?;
        let (c_values, c_valid) = parts(&c);
        let (eq_values, _) = parts(&eq);
        // Rows this field decides: a valid ordering hit decides `true`,
        // a valid inequality decides `false`, a null decides null.
        let not_decided: BooleanBuffer = !&decided;
        let hit: BooleanBuffer = &(&c_values & &c_valid) & &not_decided;
        let neq: BooleanBuffer = &c_valid & &!&eq_values;
        let miss: BooleanBuffer = &neq & &not_decided;
        let null: BooleanBuffer = &!&c_valid & &not_decided;
        values |= &hit;
        decided |= &hit;
        decided |= &miss;
        decided |= &null;
        nulls |= &null;
        if decided.count_set_bits() == rows {
            break;
        }
    }
    Ok(BooleanArray::new(values, {
        (nulls.count_set_bits() > 0).then(|| NullBuffer::new(!&nulls))
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
fn bin_value(a: &dyn Array, i: usize) -> &[u8] {
    match a.data_type() {
        ArrowType::BinaryView => a.as_binary_view().value(i),
        ArrowType::LargeBinary => a.as_binary::<i64>().value(i),
        _ => a.as_binary::<i32>().value(i),
    }
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
    use arrow_array::{Int32Array, NullArray};

    use super::*;

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
}
