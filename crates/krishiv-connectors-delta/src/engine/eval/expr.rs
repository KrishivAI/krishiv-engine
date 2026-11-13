// SPDX-License-Identifier: Apache-2.0
//! Expression evaluation (port of the kernel's `evaluate_expression`, `delta_kernel` 0.30.1
//! `engine/arrow_expression/evaluate_expression.rs`, Apache-2.0).
//!
//! - A column under a struct is null wherever a struct above it is null (the kernel's readers
//!   fix nested null masks first; Krishiv's batches arrive unfixed).
//! - Arithmetic is arrow-arith's checked arithmetic: integer division truncates, an integer
//!   division by zero or an overflow is an error, a float division by zero is infinity or NaN.
//! - `Cast` is Arrow's safe cast: a value that does not convert (a non-numeric string, an
//!   overflow) is NULL, and a type pair Arrow cannot cast gives an all-NULL column (so an
//!   unsupported cast keeps the file in data skipping).
//! - `Coalesce` stops evaluating at the first input with no nulls; its inputs must share one
//!   type.
//! - `Struct` takes the output field names and nullability from the result type; its optional
//!   nullability predicate makes a row null unless the predicate is true.
//! - A column whose result type is a struct has its children picked by name in the declared
//!   order (the kernel pairs them by position after a name-based check, which would swap the
//!   values of a struct stored in another field order).
//! - `StructPatch` walks the input struct's fields in order: kept fields, then each field's
//!   insertions, then the appended ones; a non-optional patch naming no input field is an
//!   error, whatever the optional patches match; the input struct's nulls are kept.
//! - `Opaque` and `Unknown` are unsupported: the kernel turns an unknown expression into NULL
//!   for data skipping itself, and an engine must never read it as NULL in a filter.

use std::sync::Arc;

use arrow_arith::numeric::{add, div, mul, sub};
use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BooleanArray, ListArray, RecordBatch, StructArray, make_array, new_null_array,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_cast::can_cast_types;
use arrow_data::transform::MutableArrayData;
use arrow_schema::{Field as ArrowField, Fields};
use delta_kernel::expressions::{
    BinaryExpression, BinaryExpressionOp, Expression, ExpressionRef, ExpressionStructPatch,
    UnaryExpression, UnaryExpressionOp, VariadicExpression, VariadicExpressionOp,
};
use delta_kernel::schema::{DataType, StructType};
use delta_kernel::{KernelError, KernelResult};

use super::predicate::evaluate_predicate;
use super::types::{cast_to, check, select_by_name, validate};
use super::{Ctx, arrow_err, json, partition};
use crate::engine::schema::{scalar_to_array, to_arrow_type};

/// The list element field name of the kernel's Arrow conversion.
const LIST_ELEMENT: &str = "element";

/// The column at the dotted `path` of `batch`, null wherever one of the structs above it is
/// null.
///
/// Arrow lets a null struct row hold a non-null child value; the kernel's evaluator reads
/// children as they are because its readers first apply `fix_nested_null_masks`. Krishiv's
/// batches come unfixed, so the ancestors' validity is applied here, as the getters in
/// `engine/data.rs` do: `s.x = 7` is NULL for a NULL `s`, whatever `x` physically holds.
///
/// # Errors
/// An empty path, a missing field, or a step through a non-struct.
pub(crate) fn extract_column(batch: &RecordBatch, path: &[String]) -> KernelResult<ArrayRef> {
    let mut names = path.iter();
    let first = names
        .next()
        .ok_or_else(|| KernelError::generic("Empty column path"))?;
    let mut column = batch
        .column_by_name(first)
        .ok_or_else(|| KernelError::generic(format!("No such field: {first}")))?;
    let mut ancestors: Option<NullBuffer> = None;
    for name in names {
        let parent = column
            .as_struct_opt()
            .ok_or_else(|| KernelError::generic(format!("Not a struct: {name}")))?;
        ancestors = NullBuffer::union(ancestors.as_ref(), parent.nulls());
        column = parent
            .column_by_name(name)
            .ok_or_else(|| KernelError::generic(format!("No such field: {name}")))?;
    }
    match ancestors {
        Some(nulls) if nulls.null_count() > 0 => with_parent_nulls(column, &nulls),
        _ => Ok(Arc::clone(column)),
    }
}

/// `column` with `parents` (the combined validity of the structs above it) unioned into its
/// own nulls and, for a struct, into every descendant's (the kernel's
/// `compute_nested_null_masks`). No values are copied.
fn with_parent_nulls(column: &ArrayRef, parents: &NullBuffer) -> KernelResult<ArrayRef> {
    // A `Null` column has no null buffer and is all null already.
    if column.data_type() == &arrow_schema::DataType::Null {
        return Ok(Arc::clone(column));
    }
    let nulls = NullBuffer::union(Some(parents), column.nulls());
    if let Some(s) = column.as_struct_opt() {
        let columns = match &nulls {
            Some(n) => s
                .columns()
                .iter()
                .map(|c| with_parent_nulls(c, n))
                .collect::<KernelResult<Vec<_>>>()?,
            None => s.columns().to_vec(),
        };
        return build_struct(s.fields().clone(), columns, nulls, s.len());
    }
    let data = column
        .to_data()
        .into_builder()
        .nulls(nulls)
        .build()
        .map_err(arrow_err)?;
    Ok(make_array(data))
}

/// Evaluates `expression` over `batch`; `result_type`, when given, is checked.
///
/// # Errors
/// The kernel's evaluation errors (see the module docs).
pub(crate) fn evaluate_expression(
    ctx: &Ctx,
    expression: &Expression,
    batch: &RecordBatch,
    result_type: Option<&DataType>,
) -> KernelResult<ArrayRef> {
    match (expression, result_type) {
        (Expression::Literal(scalar), _) => {
            validate(scalar_to_array(scalar, batch.num_rows())?, result_type)
        }
        (Expression::Column(name), _) => {
            let column = validate(extract_column(batch, name.path())?, result_type)?;
            match result_type {
                Some(dt) => select_by_name(column, dt),
                None => Ok(column),
            }
        }
        (Expression::Struct(fields, nullability), Some(DataType::Struct(schema))) => {
            evaluate_struct(ctx, fields, nullability.as_ref(), batch, schema)
        }
        (Expression::Struct(..), dt) => Err(KernelError::generic(format!(
            "Struct expression expects a DataType::Struct result, but got {dt:?}"
        ))),
        (Expression::StructPatch(patch), Some(DataType::Struct(schema))) => {
            evaluate_struct_patch(ctx, patch, batch, schema)
        }
        (Expression::StructPatch(_), _) => Err(KernelError::generic(
            "Data type is required to evaluate struct patch expressions",
        )),
        (Expression::Predicate(pred), None | Some(&DataType::BOOLEAN)) => {
            Ok(Arc::new(evaluate_predicate(ctx, pred, batch, false)?))
        }
        (Expression::Predicate(_), Some(dt)) => Err(KernelError::generic(format!(
            "Predicate evaluation produces boolean output, but caller expects {dt:?}"
        ))),
        (
            Expression::Unary(UnaryExpression {
                op: UnaryExpressionOp::ToJson,
                expr,
            }),
            None | Some(&DataType::STRING),
        ) => {
            let input = evaluate_expression(ctx, expr, batch, None)?;
            json::to_json(&input)
        }
        (Expression::Unary(_), Some(dt)) => Err(KernelError::generic(format!(
            "ToJson operator requires STRING output, but got {dt:?}"
        ))),
        (Expression::Binary(BinaryExpression { op, left, right }), _) => {
            let l = evaluate_expression(ctx, left, batch, None)?;
            let r = evaluate_expression(ctx, right, batch, None)?;
            let out = match op {
                BinaryExpressionOp::Plus => add(&l, &r),
                BinaryExpressionOp::Minus => sub(&l, &r),
                BinaryExpressionOp::Multiply => mul(&l, &r),
                BinaryExpressionOp::Divide => div(&l, &r),
            }
            .map_err(arrow_err)?;
            validate(out, result_type)
        }
        (
            Expression::Variadic(VariadicExpression {
                op: VariadicExpressionOp::Coalesce,
                exprs,
            }),
            _,
        ) => evaluate_coalesce(ctx, exprs, batch, result_type),
        (
            Expression::Variadic(VariadicExpression {
                op: VariadicExpressionOp::Array,
                exprs,
            }),
            _,
        ) => evaluate_array(ctx, exprs, batch, result_type),
        (Expression::Opaque(op), _) => Err(KernelError::unsupported(format!(
            "Unsupported opaque expression: {:?}",
            op.op
        ))),
        (Expression::ParseJson(p), _) => {
            let strings = evaluate_expression(ctx, &p.json_expr, batch, Some(&DataType::STRING))?;
            json::parse_json(ctx, &strings, &p.output_schema)
        }
        (Expression::MapToStruct(m), Some(DataType::Struct(schema))) => {
            let map = evaluate_expression(ctx, &m.map_expr, batch, None)?;
            partition::map_to_struct(&map, schema, &m.options)
        }
        (Expression::MapToStruct(_), dt) => Err(KernelError::generic(format!(
            "MapToStruct expression requires a DataType::Struct result type, but got {dt:?}"
        ))),
        (Expression::Cast(c), _) => {
            let input = evaluate_expression(ctx, &c.expr, batch, None)?;
            let target = to_arrow_type(&c.target)?;
            // Arrow errors on a type pair it cannot cast, rather than per value; that becomes an
            // all-NULL column, so an unsupported cast keeps the file.
            let output = if can_cast_types(input.data_type(), &target) {
                cast_to(&input, &target, true)?
            } else {
                new_null_array(&target, input.len())
            };
            validate(output, result_type)
        }
        (Expression::Unknown(name), _) => Err(KernelError::unsupported(format!(
            "Unknown expression: {name:?}"
        ))),
    }
}

fn evaluate_struct(
    ctx: &Ctx,
    fields: &[ExpressionRef],
    nullability: Option<&ExpressionRef>,
    batch: &RecordBatch,
    schema: &StructType,
) -> KernelResult<ArrayRef> {
    if fields.len() != schema.num_fields() {
        return Err(KernelError::generic(format!(
            "Struct expression field count mismatch: {} fields in expression but {} in schema",
            fields.len(),
            schema.num_fields()
        )));
    }
    let columns = fields
        .iter()
        .zip(schema.fields())
        .map(|(e, f)| evaluate_expression(ctx, e, batch, Some(f.data_type())))
        .collect::<KernelResult<Vec<_>>>()?;
    let arrow_fields: Fields = columns
        .iter()
        .zip(schema.fields())
        .map(|(c, f)| ArrowField::new(f.name(), c.data_type().clone(), f.is_nullable()))
        .collect();
    let nulls = match nullability {
        None => None,
        Some(pred) => {
            let keep = evaluate_expression(ctx, pred, batch, Some(&DataType::BOOLEAN))?;
            let keep = keep.as_boolean_opt().ok_or_else(|| {
                KernelError::generic("Nullability predicate must evaluate to boolean")
            })?;
            let values = match keep.nulls() {
                Some(n) => keep.values() & n.inner(),
                None => keep.values().clone(),
            };
            Some(NullBuffer::new(values))
        }
    };
    build_struct(arrow_fields, columns, nulls, batch.num_rows())
}

fn build_struct(
    fields: Fields,
    columns: Vec<ArrayRef>,
    nulls: Option<NullBuffer>,
    rows: usize,
) -> KernelResult<ArrayRef> {
    if columns.is_empty() {
        return Ok(Arc::new(StructArray::new_empty_fields(rows, nulls)));
    }
    Ok(Arc::new(
        StructArray::try_new(fields, columns, nulls).map_err(arrow_err)?,
    ))
}

fn evaluate_struct_patch(
    ctx: &Ctx,
    patch: &ExpressionStructPatch,
    batch: &RecordBatch,
    schema: &StructType,
) -> KernelResult<ArrayRef> {
    let mut out_types = schema
        .fields()
        .map(delta_kernel::schema::StructField::data_type);
    let mut next_type = || {
        out_types
            .next()
            .ok_or_else(|| KernelError::generic("Too few fields in output schema"))
    };
    let mut columns = Vec::with_capacity(schema.num_fields());
    for e in &patch.prepended_fields {
        columns.push(evaluate_expression(ctx, e, batch, Some(next_type()?))?);
    }
    let source = patch
        .input_path()
        .map(|path| extract_column(batch, path.path()))
        .transpose()?;
    let source_struct = match &source {
        Some(a) => Some(
            a.as_struct_opt()
                .ok_or_else(|| KernelError::generic("Input path must point to a struct"))?,
        ),
        None => None,
    };
    let input_fields: &Fields = match source_struct {
        Some(s) => s.fields(),
        None => batch.schema_ref().fields(),
    };
    let mut used = 0;
    for (i, field) in input_fields.iter().enumerate() {
        let name = field.name().as_str();
        let field_patch = patch.field_patches.get(name);
        if field_patch.is_none_or(|p| p.keep_input) {
            columns.push(match source_struct {
                Some(s) => Arc::clone(s.column(i)),
                None => Arc::clone(batch.column(i)),
            });
            next_type()?;
        }
        if let Some(p) = field_patch {
            for e in &p.insertions {
                columns.push(evaluate_expression(ctx, e, batch, Some(next_type()?))?);
            }
            if !p.optional {
                used += 1;
            }
        }
    }
    let required = patch.field_patches.values().filter(|p| !p.optional).count();
    if used < required {
        return Err(KernelError::generic(
            "Some non-optional field patches reference invalid input field names",
        ));
    }
    for e in &patch.appended_fields {
        columns.push(evaluate_expression(ctx, e, batch, Some(next_type()?))?);
    }
    if out_types.next().is_some() {
        return Err(KernelError::generic("Too many fields in output schema"));
    }
    let fields: Fields = columns
        .iter()
        .zip(schema.fields())
        .map(|(c, f)| ArrowField::new(f.name(), c.data_type().clone(), c.is_nullable()))
        .collect();
    let nulls = source_struct.and_then(|s| s.nulls().cloned());
    build_struct(fields, columns, nulls, batch.num_rows())
}

fn evaluate_coalesce(
    ctx: &Ctx,
    exprs: &[Expression],
    batch: &RecordBatch,
    result_type: Option<&DataType>,
) -> KernelResult<ArrayRef> {
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(exprs.len());
    for e in exprs {
        let a = evaluate_expression(ctx, e, batch, result_type)?;
        let done = a.null_count() == 0;
        arrays.push(a);
        if done {
            break;
        }
    }
    let Some((first, rest)) = arrays.split_first() else {
        return Err(KernelError::generic(
            "The default engine currently does not support empty COALESCE statements",
        ));
    };
    if let Some(rt) = result_type {
        let want = to_arrow_type(rt)?;
        if first.data_type() != &want {
            return Err(KernelError::generic(format!(
                "Requested result type {want:?} does not match arrays' data type {:?}",
                first.data_type()
            )));
        }
    }
    if rest.is_empty() {
        return Ok(Arc::clone(first));
    }
    for (i, a) in rest.iter().enumerate() {
        if a.len() != first.len() {
            return Err(KernelError::generic(format!(
                "Array at index {} has length {}, expected {}",
                i + 1,
                a.len(),
                first.len()
            )));
        }
        if a.data_type() != first.data_type() {
            return Err(KernelError::generic(format!(
                "Array at index {} has type {:?}, but expected {:?}",
                i + 1,
                a.data_type(),
                first.data_type()
            )));
        }
    }
    let data: Vec<_> = arrays.iter().map(Array::to_data).collect();
    let mut out = MutableArrayData::new(data.iter().collect(), true, first.len());
    // Runs of rows taken from the same input are copied at once.
    let mut row = 0;
    while row < first.len() {
        let pick = arrays.iter().position(|a| a.is_valid(row));
        let mut end = row + 1;
        while end < first.len() && arrays.iter().position(|a| a.is_valid(end)) == pick {
            end += 1;
        }
        match pick {
            Some(i) => out.try_extend(i, row, end).map_err(arrow_err)?,
            None => out.try_extend_nulls(end - row).map_err(arrow_err)?,
        }
        row = end;
    }
    Ok(make_array(out.freeze()))
}

fn evaluate_array(
    ctx: &Ctx,
    exprs: &[Expression],
    batch: &RecordBatch,
    result_type: Option<&DataType>,
) -> KernelResult<ArrayRef> {
    let rows = batch.num_rows();
    let array_type = match result_type {
        Some(DataType::Array(a)) => Some(a.as_ref()),
        Some(other) => {
            return Err(KernelError::generic(format!(
                "Array expression requires a DataType::Array result type, but got {other:?}"
            )));
        }
        None => None,
    };
    let element_type = array_type.map(delta_kernel::schema::ArrayType::element_type);
    let contains_null = array_type.is_none_or(delta_kernel::schema::ArrayType::contains_null);
    let elements = exprs
        .iter()
        .map(|e| evaluate_expression(ctx, e, batch, element_type))
        .collect::<KernelResult<Vec<_>>>()?;
    let arrow_element = elements
        .first()
        .ok_or_else(|| KernelError::generic("Array expression requires at least one element"))?
        .data_type()
        .clone();
    for (i, a) in elements.iter().enumerate() {
        if a.data_type() != &arrow_element {
            return Err(KernelError::generic(format!(
                "Array expression inputs must share the same element type; input 0 evaluates \
                 to {arrow_element:?} but input {i} evaluates to {:?}",
                a.data_type()
            )));
        }
        if !contains_null && a.null_count() > 0 {
            return Err(KernelError::generic(format!(
                "Array expression declares non-nullable elements (result_type contains_null \
                 is false) but input {i} contains {} null value(s)",
                a.null_count()
            )));
        }
    }
    let n = elements.len();
    let total = rows.checked_mul(n).ok_or_else(|| {
        KernelError::generic(format!(
            "Array expression length overflows usize: num_rows={rows} * inputs={n}"
        ))
    })?;
    if i32::try_from(total).is_err() {
        return Err(KernelError::generic(format!(
            "Array expression offsets overflow i32: num_rows={rows} * inputs={n}; \
             LargeListArray would be required"
        )));
    }
    let data: Vec<_> = elements.iter().map(Array::to_data).collect();
    let mut values = MutableArrayData::new(data.iter().collect(), false, total);
    for row in 0..rows {
        for col in 0..n {
            values.try_extend(col, row, row + 1).map_err(arrow_err)?;
        }
    }
    let values = make_array(values.freeze());
    let offsets = OffsetBuffer::<i32>::from_lengths(std::iter::repeat_n(n, rows));
    let field = Arc::new(ArrowField::new(LIST_ELEMENT, arrow_element, contains_null));
    let list = ListArray::try_new(field, offsets, values, None).map_err(arrow_err)?;
    if let Some(rt) = result_type {
        check(rt, list.data_type())?;
    }
    Ok(Arc::new(list))
}

/// `rows` copies of `value`.
pub(crate) fn repeat_bool(value: bool, rows: usize) -> BooleanArray {
    BooleanArray::from(vec![value; rows])
}
