// SPDX-License-Identifier: Apache-2.0
//! Higher-order functions over arrays and maps: `transform`, `filter`,
//! `exists` and `forall`; the folds `aggregate` and `reduce`; `zip_with`;
//! the map functions `map_filter`, `transform_keys`, `transform_values`
//! and `map_zip_with`; and `array_sort` with and without a comparator.
//!
//! The lambda bodies compile once and evaluate over flattened batches,
//! then the kernels reshape (see `krishiv_kernels::list` and
//! `krishiv_kernels::map`). The folds and the sorts evaluate their bodies
//! per row or per comparison instead: the accumulator of a fold depends on
//! the previous element, and a sort's comparisons are data-dependent, so
//! neither vectorizes. Every evaluation still works on list offsets and
//! array slices, never one Rust value per row.
//!
//! For a batch of `rows` rows whose lists hold `total` elements, the
//! flattened batch has `total` rows: the lambda's element columns are the
//! lists' child values, its index column (two-variable `transform`) counts
//! positions within each list, and every captured outer column repeats its
//! row's value per element (through the lists' offsets). A vectorized body
//! runs once over that batch — never one evaluation per element — and the
//! result rebuilds the lists with the original offsets (NULL lists stay
//! NULL).

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array, ListArray, MapArray, RecordBatch, UInt32Array};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_common::error::{ComparatorReturnsNull, KrishivError};
use krishiv_kernels::order::{cmp_rows, render_value};
use krishiv_kernels::{list, map};
use krishiv_plan::{AttrId, BoundColumn, Expr, ExprNode, ResolvedExpr, ResolvedFunction};

use super::{Input, PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::{internal, unsupported};

/// Whether `compile` handles `name` here (the analyzer's higher-order
/// functions over arrays and maps).
pub(super) fn is_supported(name: &str) -> bool {
    matches!(
        name,
        "transform"
            | "filter"
            | "exists"
            | "forall"
            | "aggregate"
            | "reduce"
            | "zip_with"
            | "map_filter"
            | "map_zip_with"
            | "transform_keys"
            | "transform_values"
            | "array_sort"
    )
}

/// Compiles the higher-order call `f` (see the module doc).
///
/// # Errors
/// An engine error when the plan is not what the analyzer resolves.
pub(super) fn compile(
    name: &str,
    f: &ResolvedFunction,
    input: Input<'_>,
    _e: &Expr,
) -> Result<Arc<dyn PhysExpr>> {
    match name {
        "transform" | "filter" | "exists" | "forall" => compile_unary(name, f, input),
        "aggregate" | "reduce" => compile_aggregate(name, f, input),
        "zip_with" => compile_zip_with(name, f, input),
        "map_filter" | "transform_keys" | "transform_values" => compile_map_unary(name, f, input),
        "map_zip_with" => compile_map_zip_with(name, f, input),
        "array_sort" => compile_sort(name, f, input),
        _ => Err(unsupported(&format!("higher-order function {name}"))),
    }
}

/// The lambda of a call with `values` value arguments and one lambda at
/// `lambda`: the parameters and body, or an engine error.
fn lambda_of(
    f: &ResolvedFunction,
    values: usize,
    lambda: usize,
) -> Result<(&[BoundColumn], &Arc<Expr>)> {
    if f.args.len() != values + 1 {
        return Err(internal(
            "a higher-order call without its values and lambda",
        ));
    }
    match &f.args[lambda].node {
        ExprNode::Lambda { params, body } => Ok((params, body)),
        _ => Err(internal("a higher-order call without a lambda")),
    }
}

/// The columns a lambda body reads: root columns and enclosing calls'
/// lambda variables (each level carries what the levels below need), and
/// this call's value and index variables.
#[derive(Default)]
struct Refs {
    /// Captured root columns, in first-use order.
    outer: Vec<AttrId>,
    /// Enclosing calls' lambda variables, in first-use order.
    outer_lambda: Vec<BoundColumn>,
    /// This call's value variables, in first-use order.
    lambda: Vec<AttrId>,
    /// This call's index variable, if used.
    index: Option<AttrId>,
}

/// Collects `refs` from `body` against `params`, the current call's
/// parameters with their stable identities (a nested call collects its own
/// variables when it compiles: here its parameters only shadow, through
/// `below`). The leading `skip` parameters are bound at evaluation (a
/// fold's accumulator), never flattened; the parameter at `index_at` is
/// the positional index, not a column. A variable of an enclosing call is
/// an outer reference here, carried through this level's flattened batch.
/// Matching is by [`AttrId`], never by name, so case sensitivity cannot
/// change it.
fn collect(
    body: &Arc<Expr>,
    params: &[BoundColumn],
    below: &mut Vec<AttrId>,
    refs: &mut Refs,
    skip: usize,
    index_at: Option<usize>,
) {
    match &body.node {
        ExprNode::Lambda {
            params: inner,
            body: nested,
        } => {
            // The nested call collects its own variables when it compiles:
            // this pass only gathers what the current level carries, so its
            // parameters stay in scope and the nested ones shadow them.
            below.extend(inner.iter().map(|p| p.id));
            collect(nested, params, below, refs, skip, index_at);
            below.truncate(below.len() - inner.len());
        }
        ExprNode::Resolved(ResolvedExpr::LambdaVar(c)) => {
            if below.contains(&c.id) {
                // A variable a nested call binds is that call's own: its
                // flattening carries it, never this one.
            } else if params.iter().take(skip).any(|p| p.id == c.id) {
                // Bound at evaluation (a fold's accumulator, a sort's
                // pair): neither flattened nor carried.
            } else if let Some(i) = params.iter().skip(skip).position(|p| p.id == c.id) {
                if Some(skip + i) == index_at {
                    if refs.index.is_none() {
                        refs.index = Some(c.id);
                    }
                } else if !refs.lambda.contains(&c.id) {
                    refs.lambda.push(c.id);
                }
            } else {
                // Bound outside the current call: carried by this level's
                // flattened batch.
                if !refs.outer_lambda.iter().any(|v| v.id == c.id) {
                    refs.outer_lambda.push(c.clone());
                }
            }
        }
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            if !refs.outer.contains(&c.id) {
                refs.outer.push(c.id);
            }
        }
        _ => body.for_each_child(&mut |c| collect(c, params, below, refs, skip, index_at)),
    }
}

/// Where one flattened-batch column comes from: every source repeats one
/// value per element of its row.
#[derive(Debug, Clone, Copy)]
enum Source {
    /// The input batch's column (a root column, or an enclosing flattened
    /// batch's column for a nested call), repeated per element.
    Root(usize),
    /// The `n`th flattened value column (element, key, merged side).
    Element(usize),
    /// Positions within each list.
    Index,
    /// A fold's running accumulator (one row per evaluation).
    Acc,
}

/// A count that does not fit its carrier.
fn too_many(n: usize) -> KrishivError {
    internal(&format!("a higher-order batch of {n} rows"))
}

/// The flattened batch's empty shape: a body that reads nothing (a
/// constant lambda) still needs `total` rows to evaluate over, and ignores
/// the column.
fn empty_flat(total: usize) -> Result<RecordBatch> {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("c0", ArrowType::Null, true)])),
        vec![Arc::new(arrow_array::NullArray::new(total)) as ArrayRef],
    )
    .map_err(|e| internal(&e.to_string()))
}

/// The flattened batch of `columns`: each column reads positionally, so the
/// fields only name positions.
fn flat_record(columns: Vec<ArrayRef>) -> Result<RecordBatch> {
    RecordBatch::try_new(
        Arc::new(Schema::new(
            columns
                .iter()
                .enumerate()
                .map(|(i, c)| Field::new(format!("c{i}"), c.data_type().clone(), true))
                .collect::<Vec<_>>(),
        )),
        columns,
    )
    .map_err(|e| internal(&e.to_string()))
}

/// The flattened batch's layout, schema and sources for `refs`: the
/// captured outer columns, then this call's value variables and index.
/// `param_types` holds each parameter's type and nullability in parameter
/// order (past `skip`), `acc` the accumulator's, and `index` whether the
/// batch carries positions. A scalar subquery reads the root batch, which
/// a nested call no longer sees, so bodies may not hold one.
fn flat_parts(
    input: Input<'_>,
    refs: &Refs,
    params: &[BoundColumn],
    param_types: &[(ArrowType, bool)],
    skip: usize,
    acc: Option<(AttrId, ArrowType, bool)>,
    index: bool,
) -> Result<(Vec<AttrId>, Arc<Schema>, Vec<Source>)> {
    // A scalar subquery reads the root batch, which a nested call no
    // longer sees, so bodies may not hold one.
    if refs
        .outer
        .iter()
        .any(|id| input.subqueries.iter().any(|(a, _)| a == id))
    {
        return Err(unsupported("scalar subqueries in lambda functions"));
    }
    let mut layout: Vec<AttrId> = Vec::new();
    let mut fields: Vec<Field> = Vec::new();
    let mut sources: Vec<Source> = Vec::new();
    for id in &refs.outer {
        let index = input
            .layout
            .iter()
            .position(|a| a == id)
            .ok_or_else(|| internal("a higher-order capture outside its input"))?;
        let field = input.schema.field(index);
        layout.push(*id);
        // The real nullability: bindings of nested values read it.
        fields.push(Field::new(
            format!("c{}", fields.len()),
            field.data_type().clone(),
            field.is_nullable(),
        ));
        sources.push(Source::Root(index));
    }
    for var in &refs.outer_lambda {
        let index = input
            .layout
            .iter()
            .position(|a| *a == var.id)
            .ok_or_else(|| internal("an enclosing lambda variable outside its input"))?;
        layout.push(var.id);
        fields.push(Field::new(
            format!("c{}", fields.len()),
            var.data_type.to_arrow(),
            var.nullable,
        ));
        // An enclosing call flattened this variable already: like a root
        // column, it repeats per element of this call's lists.
        sources.push(Source::Root(index));
    }
    if let Some((id, ty, nullable)) = acc {
        layout.push(id);
        fields.push(Field::new(format!("c{}", fields.len()), ty, nullable));
        sources.push(Source::Acc);
    }
    for id in &refs.lambda {
        let position = params
            .iter()
            .position(|p| p.id == *id)
            .ok_or_else(|| internal("a higher-order variable outside its lambda"))?;
        // `param_types` runs in parameter order (past `skip` the value
        // columns start); the flattened columns pack from zero.
        let (ty, nullable) = param_types
            .get(position)
            .ok_or_else(|| internal("a higher-order variable outside its lambda"))?;
        layout.push(*id);
        fields.push(Field::new(
            format!("c{}", fields.len()),
            ty.clone(),
            *nullable,
        ));
        sources.push(Source::Element(position.checked_sub(skip).ok_or_else(
            || internal("a higher-order variable outside its lambda"),
        )?));
    }
    if index && let Some(id) = refs.index {
        layout.push(id);
        fields.push(Field::new(
            format!("c{}", fields.len()),
            ArrowType::Int32,
            false,
        ));
        sources.push(Source::Index);
    }
    Ok((layout, Arc::new(Schema::new(fields)), sources))
}

/// Compiles `body` against the flattened layout.
fn compile_body(
    body: &Arc<Expr>,
    layout: &[AttrId],
    schema: &Arc<Schema>,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    super::compile(
        body,
        Input {
            layout,
            schema,
            env: input.env,
            subqueries: input.subqueries,
        },
    )
}

/// Rebases a sliced list to zero-based offsets: a slice keeps the absolute
/// offsets, so `offsets[0]` may be non-zero while the kernels see
/// zero-based lists. Returns the zero-based offsets with the slice's base
/// and end.
fn rebased_offsets(offsets: &[i32], rows: usize) -> Result<(Vec<i32>, usize, usize)> {
    if offsets.len() != rows + 1 {
        return Err(internal("a higher-order list of the wrong row count"));
    }
    let base = usize::try_from(offsets[0])
        .map_err(|_| internal("a higher-order list with a negative offset"))?;
    let end = usize::try_from(offsets[rows])
        .map_err(|_| internal("a higher-order list with a negative offset"))?;
    if end < base {
        return Err(internal("a higher-order list with decreasing offsets"));
    }
    let mut norm = Vec::with_capacity(rows + 1);
    for &o in offsets.iter() {
        let v = usize::try_from(o)
            .map_err(|_| internal("a higher-order list with a negative offset"))?;
        let v = v
            .checked_sub(base)
            .ok_or_else(|| internal("a higher-order list with decreasing offsets"))?;
        norm.push(
            i32::try_from(v).map_err(|_| internal("a higher-order list with a negative offset"))?,
        );
    }
    Ok((norm, base, end))
}

/// Rebases a sliced list to zero-based offsets (a slice keeps the absolute
/// offsets, so `offsets[0]` may be non-zero while the kernels see
/// zero-based lists).
fn rebase_list(list: &ListArray, rows: usize) -> Result<ListArray> {
    let (norm, base, end) = rebased_offsets(list.value_offsets(), rows)?;
    Ok(ListArray::new(
        Arc::clone(list.value_field()),
        OffsetBuffer::new(ScalarBuffer::from(norm)),
        list.values().slice(base, end - base),
        list.nulls().cloned(),
    ))
}

/// Rebases a sliced map to zero-based offsets (see [`rebase_list`]).
fn rebase_map(map: &MapArray, rows: usize) -> Result<MapArray> {
    let (norm, base, end) = rebased_offsets(map.value_offsets(), rows)?;
    let entries = map.entries();
    let field = Arc::new(Field::new("entries", entries.data_type().clone(), false));
    MapArray::try_new(
        field,
        OffsetBuffer::new(ScalarBuffer::from(norm)),
        entries.slice(base, end - base),
        map.nulls().cloned(),
        false,
    )
    .map_err(|e| internal(&e.to_string()))
}

/// The rebuilt offsets of per-row `counts`: the row keeps its length, so
/// they only re-accumulate.
fn offsets_of(counts: &[usize]) -> Result<Vec<i32>> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0_i32);
    for n in counts {
        offsets.push(
            offsets
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(i32::try_from(*n).map_err(|_| too_many(*n))?)
                .ok_or_else(|| too_many(counts.iter().sum()))?,
        );
    }
    Ok(offsets)
}

/// The row each element belongs to: `offsets[r]..offsets[r+1]`.
fn rows_of(offsets: &[i32], rows: usize) -> Result<UInt32Array> {
    if offsets.len() != rows + 1 {
        return Err(internal("a higher-order list of the wrong row count"));
    }
    let mut at = Vec::with_capacity(
        usize::try_from(offsets[rows].max(0))
            .map_err(|_| internal("a higher-order list with a negative offset"))?,
    );
    for r in 0..rows {
        let (start, stop) = (offsets[r], offsets[r + 1]);
        if stop < start {
            return Err(internal("a higher-order list with decreasing offsets"));
        }
        let r = u32::try_from(r).map_err(|_| too_many(rows))?;
        at.extend(std::iter::repeat_n(
            r,
            usize::try_from(stop - start)
                .map_err(|_| internal("a higher-order list with decreasing offsets"))?,
        ));
    }
    Ok(UInt32Array::from(at))
}

/// A bound higher-order call over one list.
#[derive(Debug)]
struct Call {
    name: String,
    array: Arc<dyn PhysExpr>,
    body: Arc<dyn PhysExpr>,
    sources: Vec<Source>,
    out: ArrowType,
    nullable: bool,
}

/// Compiles `transform`, `filter`, `exists` and `forall` (see the module
/// doc).
fn compile_unary(name: &str, f: &ResolvedFunction, input: Input<'_>) -> Result<Arc<dyn PhysExpr>> {
    if f.args.len() != 2 {
        return Err(internal("a higher-order call without its array and lambda"));
    }
    let (params, body) = lambda_of(f, 1, 1)?;
    let array = super::compile(&f.args[0], input)?;
    let ArrowType::List(field) = array.data_type().clone() else {
        return Err(internal("a higher-order call over a non-list"));
    };
    let index = name == "transform" && params.len() == 2;
    let mut refs = Refs::default();
    collect(
        body,
        params,
        &mut Vec::new(),
        &mut refs,
        0,
        index.then_some(1),
    );
    let element = field.data_type().clone();
    let element_nullable = field.is_nullable();
    let (layout, schema, sources) = flat_parts(
        input,
        &refs,
        params,
        &[(element, element_nullable)],
        0,
        None,
        index,
    )?;
    let body = compile_body(body, &layout, &schema, input)?;
    let out = f.data_type.to_arrow();
    Ok(Arc::new(Call {
        name: name.to_owned(),
        array,
        body,
        sources,
        out,
        nullable: f.nullable,
    }))
}

impl PhysExpr for Call {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        let array = self.array.eval(batch, partition)?.into_array(rows)?;
        if array.len() != rows {
            return Err(internal("a higher-order array of the wrong row count"));
        }
        let list = array
            .as_list_opt::<i32>()
            .ok_or_else(|| internal("a higher-order array that is no list"))?;
        let list = rebase_list(list, rows)?;
        let offsets = list.value_offsets();
        let total = usize::try_from(
            offsets[rows]
                .checked_sub(offsets[0])
                .ok_or_else(|| internal("a higher-order list with decreasing offsets"))?,
        )
        .map_err(|_| internal("a higher-order list with a negative offset"))?;
        let rows_of = rows_of(offsets, rows)?;
        let flat = flat_batch(batch, &self.sources, &rows_of, &list, total)?;
        let body = self.body.eval(&flat, partition)?.into_array(total)?;
        if body.len() != total {
            return Err(internal("a higher-order body of the wrong row count"));
        }
        let out = match self.name.as_str() {
            "transform" => {
                let ArrowType::List(field) = &self.out else {
                    return Err(internal("a higher-order call over a non-list"));
                };
                Arc::new(
                    list::transform_list(&list, &body, field.is_nullable())
                        .map_err(|e| internal(&e.to_string()))?,
                ) as ArrayRef
            }
            "filter" | "exists" | "forall" => {
                let keep = body
                    .as_boolean_opt()
                    .ok_or_else(|| internal("a higher-order predicate that is no boolean array"))?;
                if self.name == "filter" {
                    Arc::new(list::filter_list(&list, keep).map_err(|e| internal(&e.to_string()))?)
                        as ArrayRef
                } else if self.name == "exists" {
                    Arc::new(list::exists_list(&list, keep).map_err(|e| internal(&e.to_string()))?)
                        as ArrayRef
                } else {
                    Arc::new(list::forall_list(&list, keep).map_err(|e| internal(&e.to_string()))?)
                        as ArrayRef
                }
            }
            _ => return Err(unsupported(&format!("higher-order function {}", self.name))),
        };
        if out.data_type() != &self.out {
            return Err(internal(&format!(
                "{} returned {} where the analyzer typed {}",
                self.name,
                out.data_type(),
                self.out
            )));
        }
        Ok(Value::Array(out))
    }
}

/// The repeated root column of the flattened batch: the row's value per
/// element of its row.
fn repeat_root(batch: &RecordBatch, index: usize, rows_of: &UInt32Array) -> Result<ArrayRef> {
    let values = Arc::clone(batch.column(index));
    arrow_select::take::take(values.as_ref(), rows_of, None).map_err(|e| internal(&e.to_string()))
}

/// The flattened batch of `total` rows: root columns repeated per element,
/// element columns sliced, and the positional index.
fn flat_batch(
    batch: &RecordBatch,
    sources: &[Source],
    rows_of: &UInt32Array,
    list: &ListArray,
    total: usize,
) -> Result<RecordBatch> {
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(sources.len());
    for source in sources {
        columns.push(match source {
            Source::Root(index) => repeat_root(batch, *index, rows_of)?,
            Source::Element(_) => {
                let offsets = list.value_offsets();
                let base = usize::try_from(offsets[0]).unwrap_or(0);
                list.values().slice(base, total)
            }
            Source::Index => {
                let mut at_vals: Vec<i32> = Vec::with_capacity(total);
                let mut prev: Option<u32> = None;
                let mut pos: i32 = 0;
                for &r in rows_of.values() {
                    if prev.replace(r) != Some(r) {
                        pos = 0;
                    }
                    at_vals.push(pos);
                    pos = pos.saturating_add(1);
                }
                Arc::new(Int32Array::from(at_vals))
            }
            Source::Acc => {
                return Err(internal("a higher-order fold outside its fold"));
            }
        });
    }
    if columns.is_empty() {
        // A body that reads nothing (a constant lambda) still needs
        // `total` rows to evaluate over; it ignores the column.
        return empty_flat(total);
    }
    flat_record(columns)
}

/// A bound fold: `aggregate` or `reduce` (see the module doc). The merge
/// body runs once per element, threading the one-row accumulator; the
/// finish body runs once per row over the final accumulator (absent, the
/// accumulator passes through).
#[derive(Debug)]
struct Fold {
    name: String,
    array: Arc<dyn PhysExpr>,
    zero: Arc<dyn PhysExpr>,
    merge: Arc<dyn PhysExpr>,
    finish: Option<Arc<dyn PhysExpr>>,
    merge_schema: Arc<Schema>,
    merge_sources: Vec<Source>,
    finish_schema: Arc<Schema>,
    finish_sources: Vec<Source>,
    out: ArrowType,
    nullable: bool,
}

/// Compiles `aggregate` and `reduce` (three or four arguments).
fn compile_aggregate(
    name: &str,
    f: &ResolvedFunction,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    if f.args.len() != 3 && f.args.len() != 4 {
        return Err(internal(
            "a higher-order call without its values and lambda",
        ));
    }
    let (merge_params, merge_body) = lambda_of(f, f.args.len() - 1, 2)?;
    if merge_params.len() != 2 {
        return Err(internal("a higher-order call without a lambda"));
    }
    let array = super::compile(&f.args[0], input)?;
    let ArrowType::List(array_field) = array.data_type().clone() else {
        return Err(internal("a higher-order call over a non-list"));
    };
    let zero = super::compile(&f.args[1], input)?;
    let zero_ty = zero.data_type().clone();
    let element = array_field.data_type().clone();
    let element_nullable = array_field.is_nullable();
    let mut merge_refs = Refs::default();
    collect(
        merge_body,
        merge_params,
        &mut Vec::new(),
        &mut merge_refs,
        1,
        None,
    );
    let (merge_layout, merge_schema, merge_sources) = flat_parts(
        input,
        &merge_refs,
        merge_params,
        &[(zero_ty.clone(), true), (element, element_nullable)],
        1,
        Some((merge_params[0].id, zero_ty, true)),
        false,
    )?;
    let merge = compile_body(merge_body, &merge_layout, &merge_schema, input)?;
    let (finish, finish_schema, finish_sources) = if f.args.len() == 4 {
        let ExprNode::Lambda {
            params: finish_params,
            body: finish_body,
        } = &f.args[3].node
        else {
            return Err(internal("a higher-order call without a lambda"));
        };
        if finish_params.len() != 1 {
            return Err(internal("a higher-order call without a lambda"));
        }
        // The analyzer fills the three-argument form with an identity
        // `finish`; it has nothing to run.
        let identity = matches!(
            &finish_body.node,
            ExprNode::Resolved(ResolvedExpr::LambdaVar(c)) if c.id == finish_params[0].id
        );
        let mut finish_refs = Refs::default();
        collect(
            finish_body,
            finish_params,
            &mut Vec::new(),
            &mut finish_refs,
            1,
            None,
        );
        let (finish_layout, finish_schema, finish_sources) = flat_parts(
            input,
            &finish_refs,
            finish_params,
            &[(zero.data_type().clone(), true)],
            1,
            Some((finish_params[0].id, zero.data_type().clone(), true)),
            false,
        )?;
        let finish = compile_body(finish_body, &finish_layout, &finish_schema, input)?;
        ((!identity).then_some(finish), finish_schema, finish_sources)
    } else {
        (None, Arc::new(Schema::empty()), Vec::new())
    };
    Ok(Arc::new(Fold {
        name: name.to_owned(),
        array,
        zero,
        merge,
        finish,
        merge_schema,
        merge_sources,
        finish_schema,
        finish_sources,
        out: f.data_type.to_arrow(),
        nullable: f.nullable,
    }))
}

impl Fold {
    /// One evaluation of `body` over the rows `active` of the root batch:
    /// the captured columns, the element at `elements` of `values` and the
    /// running accumulator, each for the active rows only.
    fn step(
        &self,
        stage: (&Arc<dyn PhysExpr>, &Arc<Schema>, &[Source]),
        batch: &RecordBatch,
        active: &[u32],
        element: Option<(&ArrayRef, &[u32])>,
        acc: &ArrayRef,
        partition: u32,
    ) -> Result<ArrayRef> {
        let (body, schema, sources) = stage;
        let rows = UInt32Array::from(active.to_vec());
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(sources.len());
        for source in sources {
            columns.push(match source {
                Source::Root(index) => {
                    arrow_select::take::take(batch.column(*index).as_ref(), &rows, None)
                        .map_err(|e| internal(&e.to_string()))?
                }
                Source::Element(_) => {
                    let (values, at) =
                        element.ok_or_else(|| internal("a higher-order fold outside its fold"))?;
                    arrow_select::take::take(values.as_ref(), &UInt32Array::from(at.to_vec()), None)
                        .map_err(|e| internal(&e.to_string()))?
                }
                Source::Acc => take_rows(acc, active)?,
                Source::Index => {
                    return Err(internal("a higher-order fold outside its fold"));
                }
            });
        }
        let one = if columns.is_empty() {
            empty_flat(active.len())?
        } else {
            RecordBatch::try_new(Arc::clone(schema), columns)
                .map_err(|e| internal(&e.to_string()))?
        };
        body.eval(&one, partition)?.into_array(active.len())
    }
}

/// The rows `at` of `array`.
fn take_rows(array: &ArrayRef, at: &[u32]) -> Result<ArrayRef> {
    arrow_select::take::take(array.as_ref(), &UInt32Array::from(at.to_vec()), None)
        .map_err(|e| internal(&e.to_string()))
}

/// `base` with its rows `at` replaced by the rows of `update`, in order.
fn scatter(base: &ArrayRef, at: &[u32], update: &ArrayRef) -> Result<ArrayRef> {
    if at.len() == base.len() {
        return Ok(Arc::clone(update));
    }
    if update.len() != at.len() {
        return Err(internal("a higher-order fold step of the wrong row count"));
    }
    let mut picks: Vec<(usize, usize)> = (0..base.len()).map(|r| (0, r)).collect();
    for (pos, &r) in at.iter().enumerate() {
        let slot = picks
            .get_mut(r as usize)
            .ok_or_else(|| internal("a higher-order fold step past its rows"))?;
        *slot = (1, pos);
    }
    arrow_select::interleave::interleave(&[base.as_ref(), update.as_ref()], &picks)
        .map_err(|e| internal(&e.to_string()))
}

impl PhysExpr for Fold {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        let array = self.array.eval(batch, partition)?.into_array(rows)?;
        if array.len() != rows {
            return Err(internal("a higher-order array of the wrong row count"));
        }
        let list = array
            .as_list_opt::<i32>()
            .ok_or_else(|| internal("a higher-order array that is no list"))?;
        let list = rebase_list(list, rows)?;
        let offsets = list.value_offsets();
        let zero = self.zero.eval(batch, partition)?.into_array(rows)?;
        if zero.len() != rows {
            return Err(internal("a higher-order zero of the wrong row count"));
        }
        // Every step is evaluated for all rows at once: step `i` threads the
        // accumulators of the rows that still have an element `i` (a fold
        // cannot vectorize along one row, since each step reads the last,
        // but rows are independent), so the evaluations number the longest
        // array, not the element count.
        let mut starts: Vec<usize> = Vec::with_capacity(rows);
        let mut lens: Vec<usize> = Vec::with_capacity(rows);
        for r in 0..rows {
            let (start, stop) = (
                usize::try_from(offsets[r])
                    .map_err(|_| internal("a higher-order list with a negative offset"))?,
                usize::try_from(offsets[r + 1])
                    .map_err(|_| internal("a higher-order list with a negative offset"))?,
            );
            starts.push(start);
            lens.push(if list.is_null(r) {
                0
            } else {
                stop.saturating_sub(start)
            });
        }
        let longest = lens.iter().copied().max().unwrap_or(0);
        let mut acc: ArrayRef = zero;
        for step in 0..longest {
            let active: Vec<u32> = (0..rows)
                .filter(|&r| lens[r] > step)
                .map(|r| u32::try_from(r).map_err(|_| too_many(r)))
                .collect::<Result<_>>()?;
            let elements: Vec<u32> = active
                .iter()
                .map(|&r| {
                    let at = starts[r as usize].saturating_add(step);
                    u32::try_from(at).map_err(|_| too_many(at))
                })
                .collect::<Result<_>>()?;
            let result = self.step(
                (&self.merge, &self.merge_schema, &self.merge_sources),
                batch,
                &active,
                Some((list.values(), &elements)),
                &acc,
                partition,
            )?;
            acc = scatter(&acc, &active, &result)?;
        }
        // The rows with a list (the others are NULL).
        let valid: Vec<u32> = (0..rows)
            .filter(|&r| list.is_valid(r))
            .map(|r| u32::try_from(r).map_err(|_| too_many(r)))
            .collect::<Result<_>>()?;
        if let Some(finish) = &self.finish {
            let result = self.step(
                (finish, &self.finish_schema, &self.finish_sources),
                batch,
                &valid,
                None,
                &acc,
                partition,
            )?;
            acc = scatter(
                &arrow_array::new_null_array(&self.out, rows),
                &valid,
                &result,
            )?;
        } else if valid.len() != rows {
            acc = scatter(
                &arrow_array::new_null_array(acc.data_type(), rows),
                &valid,
                &take_rows(&acc, &valid)?,
            )?;
        }
        let out = acc;
        if out.data_type() != &self.out {
            return Err(internal(&format!(
                "{} returned {} where the analyzer typed {}",
                self.name,
                out.data_type(),
                self.out
            )));
        }
        Ok(Value::Array(out))
    }
}

/// Appends a NULL of `len` to `array` for take-based padding (a shorter
/// side reads the padded NULL).
fn padded(array: &ArrayRef, len: usize) -> Result<ArrayRef> {
    let null = arrow_array::new_null_array(array.data_type(), len);
    arrow_select::concat::concat(&[array.as_ref(), null.as_ref()])
        .map_err(|e| internal(&e.to_string()))
}

/// Whether `a` and `b` are the same type up to nested field names and
/// nullability.
fn same_shape(a: &ArrowType, b: &ArrowType) -> bool {
    match (a, b) {
        (ArrowType::List(x), ArrowType::List(y))
        | (ArrowType::LargeList(x), ArrowType::LargeList(y)) => {
            same_shape(x.data_type(), y.data_type())
        }
        (ArrowType::Struct(x), ArrowType::Struct(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(p, q)| same_shape(p.data_type(), q.data_type()))
        }
        (ArrowType::Map(x, sx), ArrowType::Map(y, sy)) => {
            sx == sy && same_shape(x.data_type(), y.data_type())
        }
        _ => a == b,
    }
}

/// The column with the analyzer's carrier: string and binary storage
/// alternatives read as their canonical `Utf8View`/`BinaryView` (as the
/// scalar bindings read them); anything else must already match.
fn canonical(column: &ArrayRef, target: &ArrowType) -> Result<ArrayRef> {
    if column.data_type() == target {
        return Ok(Arc::clone(column));
    }
    let out = match (column.data_type(), target) {
        (ArrowType::Utf8, ArrowType::Utf8View) => Arc::new(
            column
                .as_string::<i32>()
                .iter()
                .collect::<arrow_array::StringViewArray>(),
        ) as ArrayRef,
        (ArrowType::LargeUtf8, ArrowType::Utf8View) => Arc::new(
            column
                .as_string::<i64>()
                .iter()
                .collect::<arrow_array::StringViewArray>(),
        ) as ArrayRef,
        (ArrowType::Binary, ArrowType::BinaryView) => Arc::new(
            column
                .as_binary::<i32>()
                .iter()
                .collect::<arrow_array::BinaryViewArray>(),
        ) as ArrayRef,
        (ArrowType::LargeBinary, ArrowType::BinaryView) => Arc::new(
            column
                .as_binary::<i64>()
                .iter()
                .collect::<arrow_array::BinaryViewArray>(),
        ) as ArrayRef,
        // The same type up to nested nullability and field names (the
        // analyzer unifies key types of two maps that differ only there):
        // the buffers are shared and the type is the analyzer's.
        (have, want) if same_shape(have, want) => {
            let data = column
                .to_data()
                .into_builder()
                .data_type(want.clone())
                .build()
                .map_err(|e| internal(&e.to_string()))?;
            arrow_array::make_array(data)
        }
        _ => {
            return Err(internal(&format!(
                "a higher-order map of {} where the analyzer typed {target}",
                column.data_type()
            )));
        }
    };
    if out.data_type() != target {
        return Err(internal("a higher-order map of the wrong carrier"));
    }
    Ok(out)
}

/// A bound `zip_with` (see the module doc): both sides flatten, the
/// shorter padded with NULLs, and the body runs once over the pairs.
#[derive(Debug)]
struct Zip {
    left: Arc<dyn PhysExpr>,
    right: Arc<dyn PhysExpr>,
    body: Arc<dyn PhysExpr>,
    sources: Vec<Source>,
    out: ArrowType,
    nullable: bool,
}

/// Compiles `zip_with(a1, a2, (x, y) -> ...)`.
fn compile_zip_with(
    _name: &str,
    f: &ResolvedFunction,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    if f.args.len() != 3 {
        return Err(internal(
            "a higher-order call without its values and lambda",
        ));
    }
    let (params, body) = lambda_of(f, 2, 2)?;
    if params.len() != 2 {
        return Err(internal("a higher-order call without a lambda"));
    }
    let left = super::compile(&f.args[0], input)?;
    let right = super::compile(&f.args[1], input)?;
    let (ArrowType::List(first), ArrowType::List(second)) =
        (left.data_type().clone(), right.data_type().clone())
    else {
        return Err(internal("a higher-order call over a non-list"));
    };
    let mut refs = Refs::default();
    collect(body, params, &mut Vec::new(), &mut refs, 0, None);
    let (layout, schema, sources) = flat_parts(
        input,
        &refs,
        params,
        &[
            (first.data_type().clone(), true),
            (second.data_type().clone(), true),
        ],
        0,
        None,
        false,
    )?;
    let body = compile_body(body, &layout, &schema, input)?;
    Ok(Arc::new(Zip {
        left,
        right,
        body,
        sources,
        out: f.data_type.to_arrow(),
        nullable: f.nullable,
    }))
}

impl PhysExpr for Zip {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        let left = self.left.eval(batch, partition)?.into_array(rows)?;
        let right = self.right.eval(batch, partition)?.into_array(rows)?;
        if left.len() != rows || right.len() != rows {
            return Err(internal("a higher-order array of the wrong row count"));
        }
        let (Some(first), Some(second)) = (left.as_list_opt::<i32>(), right.as_list_opt::<i32>())
        else {
            return Err(internal("a higher-order array that is no list"));
        };
        let first = rebase_list(first, rows)?;
        let second = rebase_list(second, rows)?;
        let (o1, o2) = (first.value_offsets(), second.value_offsets());
        // The pair count per row is the longer side's; a NULL on either
        // side is a NULL row.
        let mut spans: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(rows);
        let mut valid: Vec<bool> = Vec::with_capacity(rows);
        for r in 0..rows {
            let (s1, l1) = (
                usize::try_from(o1[r])
                    .map_err(|_| internal("a higher-order list with a negative offset"))?,
                usize::try_from(o1[r + 1] - o1[r])
                    .map_err(|_| internal("a higher-order list with a negative offset"))?,
            );
            let (s2, l2) = (
                usize::try_from(o2[r])
                    .map_err(|_| internal("a higher-order list with a negative offset"))?,
                usize::try_from(o2[r + 1] - o2[r])
                    .map_err(|_| internal("a higher-order list with a negative offset"))?,
            );
            spans.push((s1, l1, s2, l2));
            valid.push(first.is_valid(r) && second.is_valid(r));
        }
        let counts: Vec<usize> = spans.iter().map(|&(_, l1, _, l2)| l1.max(l2)).collect();
        let total: usize = counts.iter().sum();
        let base1 = padded(first.values(), 1)?;
        let base2 = padded(second.values(), 1)?;
        // `usize` fits `u32` here: every index is below a list offset.
        let (p1, p2) = (base1.len() - 1, base2.len() - 1);
        let mut idx1: Vec<u32> = Vec::with_capacity(total);
        let mut idx2: Vec<u32> = Vec::with_capacity(total);
        let mut rows_of: Vec<u32> = Vec::with_capacity(total);
        for (r, &(s1, l1, s2, l2)) in spans.iter().enumerate() {
            let n = counts[r];
            let ru = u32::try_from(r).map_err(|_| too_many(rows))?;
            for e in 0..n {
                idx1.push(
                    u32::try_from(if e < l1 { s1 + e } else { p1 }).map_err(|_| too_many(total))?,
                );
                idx2.push(
                    u32::try_from(if e < l2 { s2 + e } else { p2 }).map_err(|_| too_many(total))?,
                );
                rows_of.push(ru);
            }
        }
        let e1 = arrow_select::take::take(base1.as_ref(), &UInt32Array::from(idx1), None)
            .map_err(|e| internal(&e.to_string()))?;
        let e2 = arrow_select::take::take(base2.as_ref(), &UInt32Array::from(idx2), None)
            .map_err(|e| internal(&e.to_string()))?;
        let rows_of = UInt32Array::from(rows_of);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.sources.len());
        for source in &self.sources {
            columns.push(match source {
                Source::Root(index) => repeat_root(batch, *index, &rows_of)?,
                Source::Element(0) => Arc::clone(&e1),
                Source::Element(_) => Arc::clone(&e2),
                Source::Index | Source::Acc => {
                    return Err(internal("a higher-order fold outside its fold"));
                }
            });
        }
        let flat = if columns.is_empty() {
            empty_flat(total)?
        } else {
            flat_record(columns)?
        };
        let body = self.body.eval(&flat, partition)?.into_array(total)?;
        if body.len() != total {
            return Err(internal("a higher-order body of the wrong row count"));
        }
        let ArrowType::List(field) = &self.out else {
            return Err(internal("a higher-order call over a non-list"));
        };
        let out = Arc::new(ListArray::new(
            Arc::clone(field),
            OffsetBuffer::new(ScalarBuffer::from(offsets_of(&counts)?)),
            body,
            Some(NullBuffer::from(valid)),
        )) as ArrayRef;
        if out.data_type() != &self.out {
            return Err(internal(&format!(
                "zip_with returned {} where the analyzer typed {}",
                out.data_type(),
                self.out
            )));
        }
        Ok(Value::Array(out))
    }
}

/// A bound map function over one map: `map_filter`, `transform_keys` or
/// `transform_values` (see the module doc). The entries flatten once, the
/// body runs once, and the map rebuilds from the original offsets (NULL
/// maps stay NULL).
#[derive(Debug)]
struct MapUnary {
    name: String,
    map: Arc<dyn PhysExpr>,
    body: Arc<dyn PhysExpr>,
    sources: Vec<Source>,
    out: ArrowType,
    nullable: bool,
    key_type: ArrowType,
    value_type: ArrowType,
    /// Whether the input map's values may be null (the rebuilt input map
    /// carries this, not the output's).
    in_value_nullable: bool,
    value_nullable: bool,
    last_win: bool,
}

impl MapUnary {
    /// The output map's key type, or `None` when `out` is no map (an
    /// engine bug: the analyzer typed one).
    fn out_key(&self) -> Option<ArrowType> {
        map_entry(&self.out, 0).map(|(t, _)| t.clone())
    }

    /// The output map's value type and nullability.
    fn out_value(&self) -> Option<(ArrowType, bool)> {
        map_entry(&self.out, 1).map(|(t, n)| (t.clone(), n))
    }
}

/// Compiles `map_filter`, `transform_keys` and `transform_values`.
fn compile_map_unary(
    name: &str,
    f: &ResolvedFunction,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    if f.args.len() != 2 {
        return Err(internal(
            "a higher-order call without its values and lambda",
        ));
    }
    let (params, body) = lambda_of(f, 1, 1)?;
    if params.len() != 2 {
        return Err(internal("a higher-order call without a lambda"));
    }
    let map = super::compile(&f.args[0], input)?;
    let ArrowType::Map(field, _) = map.data_type().clone() else {
        return Err(internal("a higher-order call over a non-map"));
    };
    let ArrowType::Struct(entries) = field.data_type().clone() else {
        return Err(internal("a higher-order call over a non-map"));
    };
    if entries.len() != 2 {
        return Err(internal("a higher-order call over a non-map"));
    }
    let (key_type, value_type) = (
        entries[0].data_type().clone(),
        entries[1].data_type().clone(),
    );
    let in_key = key_type.clone();
    let in_value = value_type.clone();
    let in_value_nullable = entries[1].is_nullable();
    let mut refs = Refs::default();
    collect(body, params, &mut Vec::new(), &mut refs, 0, None);
    // The variables carry the analyzer's nullability (nested bodies bind
    // on it).
    let (layout, schema, sources) = flat_parts(
        input,
        &refs,
        params,
        &[
            (key_type, params[0].nullable),
            (value_type, params[1].nullable),
        ],
        0,
        None,
        false,
    )?;
    let body = compile_body(body, &layout, &schema, input)?;
    let ArrowType::Map(out_field, _) = f.data_type.to_arrow() else {
        return Err(internal("a higher-order call over a non-map"));
    };
    let ArrowType::Struct(out_entries) = out_field.data_type().clone() else {
        return Err(internal("a higher-order call over a non-map"));
    };
    let value_nullable = out_entries.get(1).is_none_or(|f| f.is_nullable());
    Ok(Arc::new(MapUnary {
        name: name.to_owned(),
        map,
        body,
        sources,
        out: f.data_type.to_arrow(),
        nullable: f.nullable,
        key_type: in_key,
        value_type: in_value,
        in_value_nullable,
        value_nullable,
        last_win: input.env.map_key_dedup_last_win,
    }))
}

impl PhysExpr for MapUnary {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        let map = self.map.eval(batch, partition)?.into_array(rows)?;
        if map.len() != rows {
            return Err(internal("a higher-order map of the wrong row count"));
        }
        let map = map
            .as_map_opt()
            .ok_or_else(|| internal("a higher-order map that is no map"))?;
        let map = rebase_map(map, rows)?;
        // The entries read with the analyzer's carriers, as the flattened
        // batch's schema and the rebuilt map assume them.
        let keys = canonical(map.entries().column(0), &self.key_type)?;
        let values = canonical(map.entries().column(1), &self.value_type)?;
        let map = map::build_map(
            &keys,
            &values,
            OffsetBuffer::new(ScalarBuffer::from(map.value_offsets().to_vec())),
            map.nulls().cloned(),
            self.in_value_nullable,
        )?;
        let offsets = map.value_offsets();
        let total = usize::try_from(
            offsets[rows]
                .checked_sub(offsets[0])
                .ok_or_else(|| internal("a higher-order map with decreasing offsets"))?,
        )
        .map_err(|_| internal("a higher-order map with a negative offset"))?;
        let base = usize::try_from(offsets[0])
            .map_err(|_| internal("a higher-order map with a negative offset"))?;
        let entries = map.entries();
        let keys = entries.column(0).slice(base, total);
        let values = entries.column(1).slice(base, total);
        let rows_of = rows_of(offsets, rows)?;
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.sources.len());
        for source in &self.sources {
            columns.push(match source {
                Source::Root(index) => repeat_root(batch, *index, &rows_of)?,
                Source::Element(0) => Arc::clone(&keys),
                Source::Element(_) => Arc::clone(&values),
                Source::Index | Source::Acc => {
                    return Err(internal("a higher-order fold outside its fold"));
                }
            });
        }
        let flat = if columns.is_empty() {
            empty_flat(total)?
        } else {
            flat_record(columns)?
        };
        let body = self.body.eval(&flat, partition)?.into_array(total)?;
        if body.len() != total {
            return Err(internal("a higher-order body of the wrong row count"));
        }
        let mut valid: Vec<bool> = Vec::with_capacity(rows);
        for r in 0..rows {
            valid.push(map.is_valid(r));
        }
        let nulls = Some(NullBuffer::from(valid));
        let out = match self.name.as_str() {
            "map_filter" => {
                let keep = body
                    .as_boolean_opt()
                    .ok_or_else(|| internal("a higher-order predicate that is no boolean array"))?;
                // A NULL mark drops its entry, as Spark does.
                let mut kept: Vec<u32> = Vec::with_capacity(total);
                let mut new_offsets = Vec::with_capacity(rows + 1);
                new_offsets.push(0_i32);
                for r in 0..rows {
                    if !map.is_valid(r) {
                        new_offsets
                            .push(i32::try_from(kept.len()).map_err(|_| too_many(kept.len()))?);
                        continue;
                    }
                    let (start, stop) = (
                        usize::try_from(offsets[r])
                            .map_err(|_| internal("a higher-order map with a negative offset"))?,
                        usize::try_from(offsets[r + 1])
                            .map_err(|_| internal("a higher-order map with a negative offset"))?,
                    );
                    for e in start..stop {
                        // Entries are rebased, so `base` is already cut.
                        let at = e - base;
                        if keep.is_valid(at) && keep.value(at) {
                            kept.push(u32::try_from(e).map_err(|_| too_many(e))?);
                        }
                    }
                    new_offsets.push(i32::try_from(kept.len()).map_err(|_| too_many(kept.len()))?);
                }
                let take = UInt32Array::from(kept);
                let keys = arrow_select::take::take(map.entries().column(0).as_ref(), &take, None)
                    .map_err(|e| internal(&e.to_string()))?;
                let values =
                    arrow_select::take::take(map.entries().column(1).as_ref(), &take, None)
                        .map_err(|e| internal(&e.to_string()))?;
                map::build_map(
                    &keys,
                    &values,
                    OffsetBuffer::new(ScalarBuffer::from(new_offsets)),
                    nulls,
                    self.value_nullable,
                )?
            }
            "transform_keys" | "transform_values" => {
                let out_key = self
                    .out_key()
                    .ok_or_else(|| internal("a higher-order call over a non-map"))?;
                let (out_value, _) = self
                    .out_value()
                    .ok_or_else(|| internal("a higher-order call over a non-map"))?;
                // Each arm returns the rebuilt entries with their offsets:
                // keys may shrink them (deduplication), values never do.
                let (new_keys, new_values, new_offsets) = if self.name == "transform_keys" {
                    // Duplicate keys raise under `EXCEPTION` and keep the
                    // last value under `LAST_WIN` (see
                    // `krishiv_kernels::map`). Values come from the original
                    // map, never the body.
                    let mut kept_all: Vec<u32> = Vec::with_capacity(total);
                    let mut taken_all: Vec<u32> = Vec::with_capacity(total);
                    let mut new_offsets = Vec::with_capacity(rows + 1);
                    new_offsets.push(0_i32);
                    for r in 0..rows {
                        if !map.is_valid(r) {
                            new_offsets.push(
                                i32::try_from(kept_all.len())
                                    .map_err(|_| too_many(kept_all.len()))?,
                            );
                            continue;
                        }
                        let (start, stop) = (
                            usize::try_from(offsets[r]).map_err(|_| {
                                internal("a higher-order map with a negative offset")
                            })?,
                            usize::try_from(offsets[r + 1]).map_err(|_| {
                                internal("a higher-order map with a negative offset")
                            })?,
                        );
                        // Flat positions within the row's run.
                        let at = start.checked_sub(base).ok_or_else(|| {
                            internal("a higher-order map with decreasing offsets")
                        })?;
                        let slice = body.slice(at, stop - start);
                        let (kept, taken) =
                            map::dedup_entries(slice.as_ref(), stop - start, self.last_win)?;
                        let shift = u32::try_from(at).map_err(|_| too_many(at))?;
                        kept_all.extend(
                            kept.iter()
                                .map(|&i| i.checked_add(shift).ok_or_else(|| too_many(at)))
                                .collect::<Result<Vec<_>>>()?,
                        );
                        taken_all.extend(
                            taken
                                .iter()
                                .map(|&i| i.checked_add(shift).ok_or_else(|| too_many(at)))
                                .collect::<Result<Vec<_>>>()?,
                        );
                        new_offsets.push(
                            i32::try_from(kept_all.len()).map_err(|_| too_many(kept_all.len()))?,
                        );
                    }
                    let keys =
                        arrow_select::take::take(body.as_ref(), &UInt32Array::from(kept_all), None)
                            .map_err(|e| internal(&e.to_string()))?;
                    let values = arrow_select::take::take(
                        values.as_ref(),
                        &UInt32Array::from(taken_all),
                        None,
                    )
                    .map_err(|e| internal(&e.to_string()))?;
                    // New keys read with the output key type; values keep
                    // the map's values under the output nullability.
                    let keys = canonical(&keys, &out_key)?;
                    let values = canonical(&values, &out_value)?;
                    (keys, values, new_offsets)
                } else {
                    let values = canonical(&body, &out_value)?;
                    // Keys and values already align per entry; only the
                    // offsets and row nulls rebuild.
                    let mut new_offsets = Vec::with_capacity(rows + 1);
                    new_offsets.push(0_i32);
                    for r in 0..rows {
                        // A NULL row keeps its entries (a null row may
                        // hold some), so the offsets stay aligned with
                        // the keys and values.
                        let len = usize::try_from(offsets[r + 1] - offsets[r])
                            .map_err(|_| internal("a higher-order map with a negative offset"))?;
                        new_offsets.push(
                            new_offsets
                                .last()
                                .copied()
                                .unwrap_or(0)
                                .checked_add(i32::try_from(len).map_err(|_| too_many(len))?)
                                .ok_or_else(|| too_many(total))?,
                        );
                    }
                    (keys, values, new_offsets)
                };
                map::build_map(
                    &new_keys,
                    &new_values,
                    OffsetBuffer::new(ScalarBuffer::from(new_offsets)),
                    nulls,
                    self.value_nullable,
                )?
            }
            _ => return Err(unsupported(&format!("higher-order function {}", self.name))),
        };
        if out.data_type() != &self.out {
            return Err(internal(&format!(
                "{} returned {} where the analyzer typed {}",
                self.name,
                out.data_type(),
                self.out
            )));
        }
        Ok(Value::Array(Arc::new(out)))
    }
}

/// A bound `map_zip_with` (see the module doc): the union of the key sets
/// in first-seen order, missing values NULL, the body once over the
/// merged entries.
#[derive(Debug)]
struct MapZip {
    left: Arc<dyn PhysExpr>,
    right: Arc<dyn PhysExpr>,
    body: Arc<dyn PhysExpr>,
    sources: Vec<Source>,
    out: ArrowType,
    nullable: bool,
    key_type: ArrowType,
    v1_type: ArrowType,
    v2_type: ArrowType,
}

/// Compiles `map_zip_with(m1, m2, (k, v1, v2) -> ...)`.
fn compile_map_zip_with(
    _name: &str,
    f: &ResolvedFunction,
    input: Input<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    if f.args.len() != 3 {
        return Err(internal(
            "a higher-order call without its values and lambda",
        ));
    }
    let (params, body) = lambda_of(f, 2, 2)?;
    if params.len() != 3 {
        return Err(internal("a higher-order call without a lambda"));
    }
    let left = super::compile(&f.args[0], input)?;
    let right = super::compile(&f.args[1], input)?;
    let (ArrowType::Map(first, _), ArrowType::Map(second, _)) =
        (left.data_type().clone(), right.data_type().clone())
    else {
        return Err(internal("a higher-order call over a non-map"));
    };
    let (ArrowType::Struct(e1), ArrowType::Struct(e2)) =
        (first.data_type().clone(), second.data_type().clone())
    else {
        return Err(internal("a higher-order call over a non-map"));
    };
    if e1.len() != 2 || e2.len() != 2 {
        return Err(internal("a higher-order call over a non-map"));
    }
    let (key, v1, v2) = (
        e1[0].data_type().clone(),
        e1[1].data_type().clone(),
        e2[1].data_type().clone(),
    );
    let key_type = key.clone();
    let mut refs = Refs::default();
    collect(body, params, &mut Vec::new(), &mut refs, 0, None);
    let (layout, schema, sources) = flat_parts(
        input,
        &refs,
        params,
        &[(key, false), (v1.clone(), true), (v2.clone(), true)],
        0,
        None,
        false,
    )?;
    let body = compile_body(body, &layout, &schema, input)?;
    Ok(Arc::new(MapZip {
        left,
        right,
        body,
        sources,
        out: f.data_type.to_arrow(),
        nullable: f.nullable,
        key_type,
        v1_type: v1,
        v2_type: v2,
    }))
}

impl PhysExpr for MapZip {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        let left = self.left.eval(batch, partition)?.into_array(rows)?;
        let right = self.right.eval(batch, partition)?.into_array(rows)?;
        if left.len() != rows || right.len() != rows {
            return Err(internal("a higher-order map of the wrong row count"));
        }
        let (Some(m1), Some(m2)) = (left.as_map_opt(), right.as_map_opt()) else {
            return Err(internal("a higher-order map that is no map"));
        };
        let m1 = rebase_map(m1, rows)?;
        let m2 = rebase_map(m2, rows)?;
        // Both sides read with the analyzer's carriers (see `canonical`).
        let k1 = canonical(m1.entries().column(0), &self.key_type)?;
        let v1 = canonical(m1.entries().column(1), &self.v1_type)?;
        let k2 = canonical(m2.entries().column(0), &self.key_type)?;
        let v2 = canonical(m2.entries().column(1), &self.v2_type)?;
        let (o1, o2) = (m1.value_offsets(), m2.value_offsets());
        // The merged entries: keys in first-seen order with each side's
        // value (NULL where the side misses the key). Appended NULLs pad
        // the takes.
        let mut matches: Vec<Vec<map::UnionEntry>> = Vec::with_capacity(rows);
        let mut counts: Vec<usize> = Vec::with_capacity(rows);
        let mut valid: Vec<bool> = Vec::with_capacity(rows);
        for r in 0..rows {
            if !m1.is_valid(r) || !m2.is_valid(r) {
                matches.push(Vec::new());
                counts.push(0);
                valid.push(false);
                continue;
            }
            let (s1, l1) = (
                usize::try_from(o1[r])
                    .map_err(|_| internal("a higher-order map with a negative offset"))?,
                usize::try_from(o1[r + 1] - o1[r])
                    .map_err(|_| internal("a higher-order map with a negative offset"))?,
            );
            let (s2, l2) = (
                usize::try_from(o2[r])
                    .map_err(|_| internal("a higher-order map with a negative offset"))?,
                usize::try_from(o2[r + 1] - o2[r])
                    .map_err(|_| internal("a higher-order map with a negative offset"))?,
            );
            let m = map::union_keys(&k1.slice(s1, l1), l1, &k2.slice(s2, l2), l2)?;
            counts.push(m.len());
            let (s1u, s2u, l1u, k1u) = (
                u32::try_from(s1).map_err(|_| too_many(s1))?,
                u32::try_from(s2).map_err(|_| too_many(s2))?,
                u32::try_from(l1).map_err(|_| too_many(l1))?,
                u32::try_from(k1.len()).map_err(|_| too_many(k1.len()))?,
            );
            matches.push(
                m.into_iter()
                    .map(|(k, a, b)| {
                        // The union indexed the concatenated slices: shift
                        // back to absolute positions in the concatenated
                        // full columns.
                        let k = if k < l1u {
                            k.checked_add(s1u)
                        } else {
                            k.checked_sub(l1u)
                                .and_then(|rel| rel.checked_add(s2u)?.checked_add(k1u))
                        }
                        .ok_or_else(|| too_many(s1 + s2))?;
                        let a = a
                            .map(|i| i.checked_add(s1u).ok_or_else(|| too_many(s1)))
                            .transpose()?;
                        let b = b
                            .map(|i| i.checked_add(s2u).ok_or_else(|| too_many(s2)))
                            .transpose()?;
                        Ok((k, a, b))
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
            valid.push(true);
        }
        let total: usize = counts.iter().sum();
        let k_all = arrow_select::concat::concat(&[k1.as_ref(), k2.as_ref()])
            .map_err(|e| internal(&e.to_string()))?;
        let v1_pad = padded(&v1, 1)?;
        let v2_pad = padded(&v2, 1)?;
        let (p1, p2) = (v1.len(), v2.len());
        let mut key_take: Vec<u32> = Vec::with_capacity(total);
        let mut v1_take: Vec<u32> = Vec::with_capacity(total);
        let mut v2_take: Vec<u32> = Vec::with_capacity(total);
        let mut rows_of: Vec<u32> = Vec::with_capacity(total);
        for (r, m) in matches.iter().enumerate() {
            let ru = u32::try_from(r).map_err(|_| too_many(rows))?;
            for &(k, a, b) in m {
                key_take.push(k);
                v1_take.push(a.map_or_else(|| u32::try_from(p1).map_err(|_| too_many(p1)), Ok)?);
                v2_take.push(b.map_or_else(|| u32::try_from(p2).map_err(|_| too_many(p2)), Ok)?);
                rows_of.push(ru);
            }
        }
        let keys = arrow_select::take::take(k_all.as_ref(), &UInt32Array::from(key_take), None)
            .map_err(|e| internal(&e.to_string()))?;
        let vals1 = arrow_select::take::take(v1_pad.as_ref(), &UInt32Array::from(v1_take), None)
            .map_err(|e| internal(&e.to_string()))?;
        let vals2 = arrow_select::take::take(v2_pad.as_ref(), &UInt32Array::from(v2_take), None)
            .map_err(|e| internal(&e.to_string()))?;
        let rows_of = UInt32Array::from(rows_of);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.sources.len());
        for source in &self.sources {
            columns.push(match source {
                Source::Root(index) => repeat_root(batch, *index, &rows_of)?,
                Source::Element(0) => Arc::clone(&keys),
                Source::Element(1) => Arc::clone(&vals1),
                Source::Element(_) => Arc::clone(&vals2),
                Source::Index | Source::Acc => {
                    return Err(internal("a higher-order fold outside its fold"));
                }
            });
        }
        let flat = if columns.is_empty() {
            empty_flat(total)?
        } else {
            flat_record(columns)?
        };
        let body = self.body.eval(&flat, partition)?.into_array(total)?;
        if body.len() != total {
            return Err(internal("a higher-order body of the wrong row count"));
        }
        let out = map::build_map(
            &keys,
            &body,
            OffsetBuffer::new(ScalarBuffer::from(offsets_of(&counts)?)),
            Some(NullBuffer::from(valid)),
            self.out_value_nullable(),
        )?;
        if out.data_type() != &self.out {
            return Err(internal(&format!(
                "map_zip_with returned {} where the analyzer typed {}",
                out.data_type(),
                self.out
            )));
        }
        Ok(Value::Array(Arc::new(out)))
    }
}

impl MapZip {
    /// Whether the output map's values may be null (the analyzer's).
    fn out_value_nullable(&self) -> bool {
        map_entry(&self.out, 1).is_none_or(|(_, n)| n)
    }
}

/// The `i`th entry field of the output map `out`: its type and nullability,
/// or `None` when `out` is no map (an engine bug: the analyzer typed one).
fn map_entry(out: &ArrowType, i: usize) -> Option<(&ArrowType, bool)> {
    match out {
        ArrowType::Map(field, _) => match field.data_type() {
            ArrowType::Struct(entries) => entries.get(i).map(|f| (f.data_type(), f.is_nullable())),
            _ => None,
        },
        _ => None,
    }
}

/// A bound `array_sort` (see the module doc): without a comparator the
/// array sorts in ascending order with NULLs last (Spark's default
/// comparator); with one, a stable merge sort over the list's offsets
/// calls it per comparison (a NULL result is a `COMPARATOR_RETURNS_NULL`
/// error, as Spark raises it).
#[derive(Debug)]
struct Sort {
    array: Arc<dyn PhysExpr>,
    comparator: Option<Arc<dyn PhysExpr>>,
    schema: Arc<Schema>,
    sources: Vec<Source>,
    out: ArrowType,
    nullable: bool,
}

/// Compiles `array_sort(arr[, (l, r) -> ...])`.
fn compile_sort(_name: &str, f: &ResolvedFunction, input: Input<'_>) -> Result<Arc<dyn PhysExpr>> {
    if f.args.len() != 1 && f.args.len() != 2 {
        return Err(internal(
            "a higher-order call without its values and lambda",
        ));
    }
    let array = super::compile(&f.args[0], input)?;
    let ArrowType::List(_) = array.data_type().clone() else {
        return Err(internal("a higher-order call over a non-list"));
    };
    if f.args.len() == 1 {
        return Ok(Arc::new(Sort {
            array,
            comparator: None,
            schema: Arc::new(Schema::empty()),
            sources: Vec::new(),
            out: f.data_type.to_arrow(),
            nullable: f.nullable,
        }));
    }
    let (params, body) = lambda_of(f, 1, 1)?;
    if params.len() != 2 {
        return Err(internal("a higher-order call without a lambda"));
    }
    let mut refs = Refs::default();
    collect(body, params, &mut Vec::new(), &mut refs, 2, None);
    // Both parameters are bound per comparison, never flattened: the
    // layout below carries the captured columns first (from `flat_parts`)
    // and the two elements after (their types ride the analyzer's
    // variables).
    let (mut layout, schema, mut sources) = flat_parts(input, &refs, params, &[], 2, None, false)?;
    let mut fields: Vec<Field> = schema.fields().iter().map(|f| f.as_ref().clone()).collect();
    for (var, at) in params.iter().take(2).zip([0, 1]) {
        layout.push(var.id);
        fields.push(Field::new(
            format!("c{}", fields.len()),
            var.data_type.to_arrow(),
            var.nullable,
        ));
        sources.push(Source::Element(at));
    }
    let schema = Arc::new(Schema::new(fields));
    let body = compile_body(body, &layout, &schema, input)?;
    Ok(Arc::new(Sort {
        array,
        comparator: Some(body),
        schema,
        sources,
        out: f.data_type.to_arrow(),
        nullable: f.nullable,
    }))
}

/// Merges the adjacent runs `left` and `right` of `perm` stably through
/// `buf`: on ties the left run's element goes first. A comparison failure
/// aborts the sort with its error.
fn merge_runs(
    perm: &mut [usize],
    buf: &mut [usize],
    left: (usize, usize),
    right: (usize, usize),
    cmp: &dyn Fn(usize, usize) -> Result<Ordering, KrishivError>,
) -> Result<()> {
    let (mut i, mut j, mut k) = (left.0, right.0, left.0);
    while i < left.1 && j < right.1 {
        if cmp(perm[i], perm[j])? != Ordering::Greater {
            buf[k] = perm[i];
            i += 1;
        } else {
            buf[k] = perm[j];
            j += 1;
        }
        k += 1;
    }
    while i < left.1 {
        buf[k] = perm[i];
        i += 1;
        k += 1;
    }
    while j < right.1 {
        buf[k] = perm[j];
        j += 1;
        k += 1;
    }
    perm[left.0..right.1].copy_from_slice(&buf[left.0..right.1]);
    Ok(())
}

/// Stably sorts `perm` by `cmp`: ascending runs (and strictly descending
/// ones, reversed) merge pairwise, so an already ordered run never moves
/// (a constant comparator leaves every array in place, as Spark's sort
/// does).
fn stable_sort(
    perm: &mut [usize],
    cmp: &dyn Fn(usize, usize) -> Result<Ordering, KrishivError>,
) -> Result<()> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < perm.len() {
        let mut j = i + 1;
        if j < perm.len() && cmp(perm[j], perm[i])? == Ordering::Less {
            // Strictly descending: extend, then reverse back to ascending.
            j += 1;
            while j < perm.len() && cmp(perm[j], perm[j - 1])? == Ordering::Less {
                j += 1;
            }
            perm[i..j].reverse();
        } else {
            while j < perm.len() && cmp(perm[j], perm[j - 1])? != Ordering::Less {
                j += 1;
            }
        }
        runs.push((i, j));
        i = j;
    }
    let mut buf = vec![0_usize; perm.len()];
    while runs.len() > 1 {
        let mut next: Vec<(usize, usize)> = Vec::with_capacity(runs.len().div_ceil(2));
        let (pairs, rest) = runs.as_chunks::<2>();
        for pair in pairs {
            merge_runs(perm, &mut buf, pair[0], pair[1], cmp)?;
            next.push((pair[0].0, pair[1].1));
        }
        next.extend(rest.iter().copied());
        runs = next;
    }
    Ok(())
}

/// Where a [`Machine`] is in the sort of one array (the sequential
/// algorithm of [`stable_sort`], one comparison at a time).
#[derive(Debug)]
enum Phase {
    /// Looking for the next run from `i`.
    Start {
        i: usize,
    },
    /// The first pair of a run at `i` is out for comparison.
    First {
        i: usize,
    },
    /// Extending the run `i..j`, strictly descending when `desc`; the pair
    /// at `j` is out for comparison.
    Extend {
        i: usize,
        j: usize,
        desc: bool,
    },
    /// Merging the pair of runs at `pair`; the heads `i` and `j` are out.
    Merge {
        pair: usize,
        i: usize,
        j: usize,
        k: usize,
    },
    Done,
}

/// One array's sort as a state machine, so every array's pending
/// comparison can be evaluated in one batch per round: [`Machine::request`]
/// names the elements to compare next and [`Machine::answer`] takes the
/// result. The order of comparisons and the outcome are those of
/// [`stable_sort`].
#[derive(Debug)]
struct Machine {
    perm: Vec<usize>,
    buf: Vec<usize>,
    runs: Vec<(usize, usize)>,
    next: Vec<(usize, usize)>,
    phase: Phase,
}

impl Machine {
    fn new(perm: Vec<usize>) -> Self {
        let n = perm.len();
        Self {
            perm,
            buf: vec![0; n],
            runs: Vec::new(),
            next: Vec::new(),
            phase: Phase::Start { i: 0 },
        }
    }

    /// The pair of elements to compare next, or `None` when sorted.
    fn request(&mut self) -> Option<(usize, usize)> {
        loop {
            match self.phase {
                Phase::Done => return None,
                Phase::Start { i } => {
                    if i >= self.perm.len() {
                        self.begin_round();
                        continue;
                    }
                    if i + 1 >= self.perm.len() {
                        self.runs.push((i, i + 1));
                        self.phase = Phase::Start { i: i + 1 };
                        continue;
                    }
                    self.phase = Phase::First { i };
                    return Some((self.perm[i + 1], self.perm[i]));
                }
                Phase::First { i } => return Some((self.perm[i + 1], self.perm[i])),
                Phase::Extend { i, j, desc } => {
                    if j >= self.perm.len() {
                        self.finish_run(i, j, desc);
                        continue;
                    }
                    return Some((self.perm[j], self.perm[j - 1]));
                }
                Phase::Merge { pair, i, j, k } => {
                    let Some(&left) = self.runs.get(2 * pair) else {
                        self.end_round();
                        continue;
                    };
                    let Some(&right) = self.runs.get(2 * pair + 1) else {
                        self.next.push(left);
                        self.end_round();
                        continue;
                    };
                    if i < left.1 && j < right.1 {
                        return Some((self.perm[i], self.perm[j]));
                    }
                    let (mut i, mut j, mut k) = (i, j, k);
                    while i < left.1 {
                        self.buf[k] = self.perm[i];
                        i += 1;
                        k += 1;
                    }
                    while j < right.1 {
                        self.buf[k] = self.perm[j];
                        j += 1;
                        k += 1;
                    }
                    self.perm[left.0..right.1].copy_from_slice(&self.buf[left.0..right.1]);
                    self.next.push((left.0, right.1));
                    self.start_pair(pair + 1);
                }
            }
        }
    }

    /// Takes the result of the comparison [`Machine::request`] named.
    fn answer(&mut self, ord: Ordering) {
        match self.phase {
            Phase::First { i } => {
                self.phase = Phase::Extend {
                    i,
                    j: i + 2,
                    desc: ord == Ordering::Less,
                };
            }
            Phase::Extend { i, j, desc } => {
                if (ord == Ordering::Less) == desc {
                    self.phase = Phase::Extend { i, j: j + 1, desc };
                } else {
                    self.finish_run(i, j, desc);
                }
            }
            Phase::Merge { pair, i, j, k } => {
                if ord != Ordering::Greater {
                    self.buf[k] = self.perm[i];
                    self.phase = Phase::Merge {
                        pair,
                        i: i + 1,
                        j,
                        k: k + 1,
                    };
                } else {
                    self.buf[k] = self.perm[j];
                    self.phase = Phase::Merge {
                        pair,
                        i,
                        j: j + 1,
                        k: k + 1,
                    };
                }
            }
            Phase::Start { .. } | Phase::Done => {}
        }
    }

    fn finish_run(&mut self, i: usize, j: usize, desc: bool) {
        if desc {
            self.perm[i..j].reverse();
        }
        self.runs.push((i, j));
        self.phase = Phase::Start { i: j };
    }

    fn begin_round(&mut self) {
        if self.runs.len() > 1 {
            self.next.clear();
            self.start_pair(0);
        } else {
            self.phase = Phase::Done;
        }
    }

    fn end_round(&mut self) {
        self.runs = std::mem::take(&mut self.next);
        self.begin_round();
    }

    fn start_pair(&mut self, pair: usize) {
        let (i, j) = match (self.runs.get(2 * pair), self.runs.get(2 * pair + 1)) {
            (Some(l), Some(r)) => (l.0, r.0),
            _ => (0, 0),
        };
        let k = i;
        self.phase = Phase::Merge { pair, i, j, k };
    }
}

impl PhysExpr for Sort {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        let array = self.array.eval(batch, partition)?.into_array(rows)?;
        if array.len() != rows {
            return Err(internal("a higher-order array of the wrong row count"));
        }
        let list = array
            .as_list_opt::<i32>()
            .ok_or_else(|| internal("a higher-order array that is no list"))?;
        let list = rebase_list(list, rows)?;
        let offsets = list.value_offsets();
        let values = list.values();
        // One permutation per row; every row keeps its length (a NULL row
        // keeps its elements, unsorted), so the offsets rebuild unchanged.
        let mut perms: Vec<Vec<usize>> = Vec::with_capacity(rows);
        for r in 0..rows {
            let (start, stop) = (
                usize::try_from(offsets[r])
                    .map_err(|_| internal("a higher-order list with a negative offset"))?,
                usize::try_from(offsets[r + 1])
                    .map_err(|_| internal("a higher-order list with a negative offset"))?,
            );
            perms.push((start..stop).collect());
        }
        if let Some(comparator) = &self.comparator {
            // The comparator over pairs: the captured row with the two
            // elements. All arrays advance together, one comparison each
            // per round, evaluated as one batch. A NULL result is Spark's
            // `COMPARATOR_RETURNS_NULL` (both values render the left
            // element, as Spark's call does).
            let mut machines: Vec<(usize, Machine)> = Vec::new();
            for (r, perm) in perms.iter_mut().enumerate() {
                if list.is_valid(r) && perm.len() > 1 {
                    machines.push((r, Machine::new(std::mem::take(perm))));
                }
            }
            loop {
                let mut asked: Vec<(usize, usize, usize)> = Vec::new();
                let mut slots: Vec<usize> = Vec::new();
                for (m, (r, machine)) in machines.iter_mut().enumerate() {
                    if let Some((a, b)) = machine.request() {
                        asked.push((*r, a, b));
                        slots.push(m);
                    }
                }
                if asked.is_empty() {
                    break;
                }
                let to_u32 = |n: usize| u32::try_from(n).map_err(|_| too_many(n));
                let roots = UInt32Array::from(
                    asked
                        .iter()
                        .map(|&(r, ..)| to_u32(r))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                let lefts = UInt32Array::from(
                    asked
                        .iter()
                        .map(|&(_, a, _)| to_u32(a))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                let rights = UInt32Array::from(
                    asked
                        .iter()
                        .map(|&(_, _, b)| to_u32(b))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                let take = |column: &ArrayRef, at: &UInt32Array| {
                    arrow_select::take::take(column.as_ref(), at, None)
                        .map_err(|e| internal(&e.to_string()))
                };
                let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.sources.len());
                for source in &self.sources {
                    columns.push(match source {
                        Source::Root(index) => take(batch.column(*index), &roots)?,
                        Source::Element(0) => take(values, &lefts)?,
                        Source::Element(_) => take(values, &rights)?,
                        Source::Index | Source::Acc => {
                            return Err(internal("a higher-order fold outside its fold"));
                        }
                    });
                }
                let pairs = RecordBatch::try_new(Arc::clone(&self.schema), columns)
                    .map_err(|e| internal(&e.to_string()))?;
                let cmp = comparator
                    .eval(&pairs, partition)?
                    .into_array(asked.len())?;
                let cmp = cmp
                    .as_primitive_opt::<arrow_array::types::Int32Type>()
                    .ok_or_else(|| internal("a comparator that is no INT"))?;
                for (n, (&m, &(_, a, _))) in slots.iter().zip(&asked).enumerate() {
                    if cmp.is_null(n) {
                        return Err(KrishivError::from(ComparatorReturnsNull {
                            first_value: render_value(values.as_ref(), a),
                            second_value: render_value(values.as_ref(), a),
                        }));
                    }
                    machines[m].1.answer(cmp.value(n).cmp(&0));
                }
            }
            for (r, machine) in machines {
                perms[r] = machine.perm;
            }
        } else {
            // Ascending with NULLs last (Spark's default comparator): NULL
            // is greater than every value.
            let one = |a: usize, b: usize| -> Result<Ordering, KrishivError> {
                match (values.is_null(a), values.is_null(b)) {
                    (true, true) => Ok(Ordering::Equal),
                    (true, false) => Ok(Ordering::Greater),
                    (false, true) => Ok(Ordering::Less),
                    (false, false) => {
                        cmp_rows(values.as_ref(), a, values.as_ref(), b).map_err(Into::into)
                    }
                }
            };
            for (r, perm) in perms.iter_mut().enumerate() {
                if list.is_valid(r) {
                    stable_sort(perm, &one)?;
                }
            }
        }
        let mut taken: Vec<u32> = Vec::new();
        for perm in &perms {
            for &e in perm {
                taken.push(u32::try_from(e).map_err(|_| too_many(e))?);
            }
        }
        let sorted = arrow_select::take::take(values.as_ref(), &UInt32Array::from(taken), None)
            .map_err(|e| internal(&e.to_string()))?;
        let ArrowType::List(field) = &self.out else {
            return Err(internal("a higher-order call over a non-list"));
        };
        let out = Arc::new(ListArray::new(
            Arc::clone(field),
            list.offsets().clone(),
            sorted,
            list.nulls().cloned(),
        )) as ArrayRef;
        if out.data_type() != &self.out {
            return Err(internal(&format!(
                "array_sort returned {} where the analyzer typed {}",
                out.data_type(),
                self.out
            )));
        }
        Ok(Value::Array(out))
    }
}
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, BooleanArray, Int32Array, ListArray, RecordBatch};
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
    use krishiv_common::error::ErrorClass;
    use krishiv_kernels::KernelCtx;
    use krishiv_plan::{
        ColumnRef, Expr, ExprNode, FunctionId, FunctionKind, ResolvedExpr, ResolvedFunction,
    };

    use super::super::{EvalEnv, Input, PhysExpr, Value, compile};
    use super::{Call, too_many};
    use crate::error::Result;

    /// A [`PhysExpr`] returning `array` for every batch.
    #[derive(Debug)]
    struct Const {
        data_type: ArrowType,
        nullable: bool,
        array: ArrayRef,
    }

    impl PhysExpr for Const {
        fn data_type(&self) -> &ArrowType {
            &self.data_type
        }

        fn nullable(&self) -> bool {
            self.nullable
        }

        fn eval(&self, _batch: &RecordBatch, _partition: u32) -> Result<Value> {
            Ok(Value::Array(Arc::clone(&self.array)))
        }
    }

    fn env() -> EvalEnv {
        EvalEnv {
            ansi: true,
            legacy_statistical_aggregate: false,
            zone: None,
            zone_name: None,
            kernels: KernelCtx::new(),
            query_start_micros: 0,
            user: Arc::from("test"),
            udfs: None,
            map_key_dedup_last_win: false,
            seed_salt: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    fn col() -> Arc<Expr> {
        Arc::new(Expr::new(ExprNode::Column(ColumnRef {
            name: vec!["a".to_owned()],
            plan_id: None,
        })))
    }

    fn lambda() -> Arc<Expr> {
        Arc::new(Expr::lambda(vec!["x".to_owned()], Expr::column(["x"])))
    }

    fn func(name: &str, args: Vec<Arc<Expr>>) -> ResolvedFunction {
        ResolvedFunction {
            id: FunctionId::Registry {
                name: name.to_owned(),
                id: 0,
            },
            kind: FunctionKind::Scalar,
            args,
            distinct: false,
            filter: None,
            data_type: krishiv_types::DataType::Null,
            nullable: true,
            deterministic: true,
        }
    }

    fn list(offsets: Vec<i32>, values: Vec<i32>) -> ArrayRef {
        Arc::new(ListArray::new(
            Arc::new(Field::new("element", ArrowType::Int32, true)),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(Int32Array::from(values)),
            None,
        ))
    }

    fn batch(rows: usize) -> RecordBatch {
        let schema: SchemaRef =
            Arc::new(Schema::new(vec![Field::new("a", ArrowType::Int32, false)]));
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![0; rows])) as ArrayRef],
        )
        .unwrap()
    }

    fn call(name: &str, array: ArrayRef, body: ArrayRef, out: ArrowType) -> Call {
        Call {
            name: name.to_owned(),
            array: Arc::new(Const {
                data_type: array.data_type().clone(),
                nullable: false,
                array,
            }),
            body: Arc::new(Const {
                data_type: body.data_type().clone(),
                nullable: true,
                array: body,
            }),
            sources: Vec::new(),
            out,
            nullable: true,
        }
    }

    #[test]
    fn malformed_calls_are_internal_errors() {
        let e = env();
        let schema: SchemaRef = Arc::new(Schema::new(Vec::<Field>::new()));
        let input = Input {
            layout: &[],
            schema: &schema,
            env: &e,
            subqueries: &[],
        };
        // Without the array and the lambda.
        let f = func("transform", vec![lambda()]);
        let err = compile(
            &Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))),
            input,
        )
        .unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::InternalError);
        assert!(
            err.to_string()
                .contains("a higher-order call without its array and lambda"),
            "{err}"
        );
        // Without a lambda as the second argument (the array never compiles).
        let f = func("transform", vec![col(), col()]);
        let err = compile(
            &Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))),
            input,
        )
        .unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::InternalError);
        assert!(
            err.to_string()
                .contains("a higher-order call without a lambda"),
            "{err}"
        );
    }

    #[test]
    fn too_many_names_the_row_count() {
        let err = too_many(5);
        assert_eq!(err.error_class(), ErrorClass::InternalError);
        assert!(err.to_string().contains('5'), "{err}");
    }

    #[test]
    fn mismatched_batches_are_internal_errors() {
        // The array holds one row where the batch holds two.
        let c = call(
            "transform",
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![1, 2])),
            ArrowType::Boolean,
        );
        let err = c.eval(&batch(2), 0).unwrap_err();
        assert!(
            err.to_string()
                .contains("a higher-order array of the wrong row count"),
            "{err}"
        );
        // The body holds three values for two elements.
        let c = call(
            "transform",
            list(vec![0, 2], vec![5, 6]),
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            ArrowType::Boolean,
        );
        let err = c.eval(&batch(1), 0).unwrap_err();
        assert!(
            err.to_string()
                .contains("a higher-order body of the wrong row count"),
            "{err}"
        );
    }

    #[test]
    fn a_transform_over_a_non_list_is_an_internal_error() {
        let c = call(
            "transform",
            list(vec![0, 2], vec![5, 6]),
            Arc::new(Int32Array::from(vec![7, 8])),
            ArrowType::Int32,
        );
        let err = c.eval(&batch(1), 0).unwrap_err();
        assert!(
            err.to_string()
                .contains("a higher-order call over a non-list"),
            "{err}"
        );
    }

    #[test]
    fn an_unknown_higher_order_function_is_unsupported() {
        let c = call(
            "bogus",
            list(vec![0, 2], vec![5, 6]),
            Arc::new(BooleanArray::from(vec![true, false])),
            ArrowType::Boolean,
        );
        let err = c.eval(&batch(1), 0).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
        assert!(
            err.to_string().contains("higher-order function bogus"),
            "{err}"
        );
    }

    #[test]
    fn a_sliced_list_rebases_to_zero_based_offsets() {
        use arrow_array::Array as _;
        use arrow_array::cast::AsArray as _;
        // `[[1, 2], [3]]` sliced to its second row: absolute offsets
        // `[2, 3]` with one element.
        let sliced = list(vec![0, 2, 3], vec![1, 2, 3]).slice(1, 1);
        let mut c = call(
            "transform",
            sliced,
            Arc::new(Int32Array::from(vec![9])),
            ArrowType::List(Arc::new(Field::new("element", ArrowType::Int32, true))),
        );
        c.sources = vec![super::Source::Element(0)];
        let out = c.eval(&batch(1), 0).unwrap().into_array(1).unwrap();
        let out = out.as_list::<i32>();
        assert_eq!(out.len(), 1);
        assert_eq!(out.value_offsets(), &[0, 1]);
        assert_eq!(
            out.values()
                .as_primitive::<arrow_array::types::Int32Type>()
                .values(),
            &[9]
        );
    }

    #[test]
    fn a_body_of_an_unexpected_type_is_an_internal_error() {
        let out = ArrowType::List(Arc::new(Field::new("element", ArrowType::Utf8, true)));
        let c = call(
            "transform",
            list(vec![0, 2], vec![5, 6]),
            Arc::new(Int32Array::from(vec![7, 8])),
            out,
        );
        let err = c.eval(&batch(1), 0).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::InternalError);
        assert!(
            err.to_string().contains("transform returned List(Int32"),
            "{err}"
        );
        assert!(
            err.to_string().contains("analyzer typed List(Utf8"),
            "{err}"
        );
    }

    /// String and binary storage alternatives read as the analyzer's view
    /// carriers; any other mismatch is an engine error.
    #[test]
    fn canonical_reads_storage_alternatives_as_views() {
        use arrow_array::{BinaryArray, LargeBinaryArray, LargeStringArray, StringArray};
        let cases: Vec<(ArrayRef, ArrowType)> = vec![
            (Arc::new(StringArray::from(vec!["a"])), ArrowType::Utf8View),
            (
                Arc::new(LargeStringArray::from(vec!["a"])),
                ArrowType::Utf8View,
            ),
            (
                Arc::new(BinaryArray::from_vec(vec![b"a"])),
                ArrowType::BinaryView,
            ),
            (
                Arc::new(LargeBinaryArray::from_vec(vec![b"a"])),
                ArrowType::BinaryView,
            ),
        ];
        for (column, target) in cases {
            let out = super::super::hof::canonical(&column, &target).unwrap();
            assert_eq!(out.data_type(), &target);
            assert_eq!(out.len(), 1);
        }
        let ints: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        assert!(super::super::hof::canonical(&ints, &ArrowType::Utf8View).is_err());
        assert!(super::super::hof::canonical(&ints, &ArrowType::Int32).is_ok());
    }

    /// A NULL row may hold elements: `array_sort` keeps them in place and
    /// still sorts the others.
    #[test]
    fn array_sort_keeps_the_elements_of_null_rows() {
        use arrow_array::Array as _;
        use arrow_array::cast::AsArray;
        use arrow_buffer::NullBuffer;
        let field = Arc::new(Field::new("item", ArrowType::Int32, true));
        let list: ArrayRef = Arc::new(ListArray::new(
            Arc::clone(&field),
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 4, 6])),
            Arc::new(Int32Array::from(vec![2, 1, 9, 8, 4, 3])),
            Some(NullBuffer::from(vec![true, false, true])),
        ));
        let out = ArrowType::List(Arc::clone(&field));
        let array = Arc::new(Const {
            data_type: out.clone(),
            nullable: true,
            array: list,
        });
        let sort = super::Sort {
            array,
            comparator: None,
            schema: Arc::new(Schema::empty()),
            sources: Vec::new(),
            out,
            nullable: true,
        };
        let batch = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &arrow_array::RecordBatchOptions::new().with_row_count(Some(3)),
        )
        .unwrap();
        let got = sort.eval(&batch, 0).unwrap().into_array(3).unwrap();
        let got = got.as_list::<i32>();
        assert!(got.is_null(1));
        let values = got.values().as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(values.values().as_ref(), [1, 2, 9, 8, 3, 4]);
    }

    /// The state machine sorts like the sequential algorithm for every
    /// input shape: ascending, descending and mixed runs, ties, singles.
    #[test]
    fn the_sort_machine_matches_the_sequential_sort() {
        let cases: Vec<Vec<i32>> = vec![
            vec![],
            vec![1],
            vec![2, 1],
            vec![1, 2, 3],
            vec![3, 2, 1],
            vec![1, 3, 2, 4, 0, 5, 5, 1],
            vec![5, 5, 5, 4, 4, 9, 1],
            (0..40).map(|i| (i * 7919) % 31).collect(),
        ];
        for case in cases {
            let cmp = |a: usize, b: usize| Ok(case[a].cmp(&case[b]));
            let mut expected: Vec<usize> = (0..case.len()).collect();
            super::stable_sort(&mut expected, &cmp).unwrap();
            let mut machine = super::Machine::new((0..case.len()).collect());
            while let Some((a, b)) = machine.request() {
                machine.answer(case[a].cmp(&case[b]));
            }
            assert_eq!(machine.perm, expected, "{case:?}");
        }
    }

    /// Key columns that differ from the analyzer's type only in nested
    /// nullability read as the analyzer's type.
    #[test]
    fn canonical_reads_nullability_variants() {
        let have = ArrowType::List(Arc::new(Field::new("item", ArrowType::Int32, true)));
        let want = ArrowType::List(Arc::new(Field::new("element", ArrowType::Int32, false)));
        let list: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", ArrowType::Int32, true)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 1])),
            Arc::new(Int32Array::from(vec![1])),
            None,
        ));
        assert_eq!(list.data_type(), &have);
        let out = super::super::hof::canonical(&list, &want).unwrap();
        assert_eq!(out.data_type(), &want);
        assert_eq!(out.len(), 1);
    }
}
