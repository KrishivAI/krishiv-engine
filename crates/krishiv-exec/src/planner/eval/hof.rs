// SPDX-License-Identifier: Apache-2.0
//! Higher-order functions over arrays: `transform`, `filter`, `exists` and
//! `forall`. The lambda body compiles once against a flattened batch, then
//! evaluates once per batch over every list element; the list kernels only
//! reshape (see `krishiv_kernels::list`).
//!
//! For a batch of `rows` rows whose lists hold `total` elements, the
//! flattened batch has `total` rows: the lambda's element column is the
//! list's child values, its index column (two-variable `transform`) counts
//! positions within each list, and every captured outer column repeats its
//! row's value per element (through the lists' offsets). The body runs
//! once over that batch — never one evaluation per element — and the
//! result rebuilds the lists with the original offsets (NULL lists stay
//! NULL).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, UInt32Array};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_kernels::list;
use krishiv_plan::{AttrId, BoundColumn, Expr, ExprNode, ResolvedExpr, ResolvedFunction};

use super::{Input, PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::{internal, unsupported};
use krishiv_common::error::KrishivError;

/// Whether `compile` handles `name` here (the analyzer's higher-order
/// functions over arrays).
pub(super) fn is_supported(name: &str) -> bool {
    matches!(name, "transform" | "filter" | "exists" | "forall")
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
    let [array_expr, lambda_expr] = f.args.as_slice() else {
        return Err(internal("a higher-order call without its array and lambda"));
    };
    let ExprNode::Lambda { params, body } = &lambda_expr.node else {
        return Err(internal("a higher-order call without a lambda"));
    };
    let array = super::compile(array_expr, input)?;
    let ArrowType::List(field) = array.data_type().clone() else {
        return Err(internal("a higher-order call over a non-list"));
    };
    let mut refs = Refs::default();
    collect(
        body,
        0,
        params,
        &mut Vec::new(),
        input.env.case_sensitive,
        &mut refs,
    );
    // The flattened batch's columns: the captured outer columns, then this
    // call's element variables and index. A scalar subquery reads the root
    // batch, which a nested call no longer sees, so bodies may not hold
    // one.
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
    for id in refs.outer {
        let index = input
            .layout
            .iter()
            .position(|a| *a == id)
            .ok_or_else(|| internal("a higher-order capture outside its input"))?;
        let field = input.schema.field(index);
        layout.push(id);
        // The real nullability: bindings of nested values read it.
        fields.push(Field::new(
            format!("c{}", fields.len()),
            field.data_type().clone(),
            field.is_nullable(),
        ));
        sources.push(Source::Root(index));
    }
    for var in refs.outer_lambda {
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
    let element = field.data_type().clone();
    let element_nullable = field.is_nullable();
    for id in refs.lambda {
        layout.push(id);
        fields.push(Field::new(
            format!("c{}", fields.len()),
            element.clone(),
            element_nullable,
        ));
        sources.push(Source::Element);
    }
    if let Some(id) = refs.index {
        layout.push(id);
        fields.push(Field::new(
            format!("c{}", fields.len()),
            ArrowType::Int32,
            false,
        ));
        sources.push(Source::Index);
    }
    let schema = Arc::new(Schema::new(fields));
    let body = super::compile(
        body,
        Input {
            layout: &layout,
            schema: &schema,
            env: input.env,
            subqueries: input.subqueries,
        },
    )?;
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

/// The columns a lambda body reads: root columns and enclosing calls'
/// lambda variables at any depth (each level carries what the levels below
/// need), and this call's element and index variables (depth 0).
#[derive(Default)]
struct Refs {
    /// Captured root columns, in first-use order.
    outer: Vec<AttrId>,
    /// Enclosing calls' lambda variables, in first-use order.
    outer_lambda: Vec<BoundColumn>,
    /// This call's element variables, in first-use order.
    lambda: Vec<AttrId>,
    /// This call's index variable, if used.
    index: Option<AttrId>,
}

/// Collects `refs` from `body` against `params`, the enclosing lambda's
/// parameters (a nested lambda collects against its own). A variable of an
/// enclosing call is an outer reference here, carried through this level's
/// flattened batch. Name matching follows `case_sensitive`, exactly like
/// the analyzer resolves it.
fn collect(
    body: &Arc<Expr>,
    depth: usize,
    params: &[String],
    below: &mut Vec<String>,
    case_sensitive: bool,
    refs: &mut Refs,
) {
    match &body.node {
        ExprNode::Lambda {
            params: inner,
            body: nested,
        } => {
            below.extend(inner.iter().cloned());
            collect(nested, depth + 1, inner, below, case_sensitive, refs);
            below.truncate(below.len() - inner.len());
        }
        ExprNode::Resolved(ResolvedExpr::LambdaVar(c)) => {
            // The innermost scope naming the variable, like the analyzer
            // resolves it: exact when case-sensitive, else a
            // case-insensitive fallback.
            let at = |scope: &[String]| {
                scope.iter().position(|p| p == &c.name).or_else(|| {
                    (!case_sensitive)
                        .then(|| scope.iter().position(|p| p.eq_ignore_ascii_case(&c.name)))
                        .flatten()
                })
            };
            if depth == 0 && at(params) == Some(0) {
                if !refs.lambda.contains(&c.id) {
                    refs.lambda.push(c.id);
                }
            } else if depth == 0 && at(params).is_some() {
                if refs.index.is_none() {
                    refs.index = Some(c.id);
                }
            } else if depth == 0 || at(below).is_none() {
                // Bound outside the current call (or the current call has
                // no such parameter): carried by this level's flattened
                // batch. A variable a nested call binds is that call's
                // own: its flattening carries it, never this one.
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
        _ => body.for_each_child(&mut |c| collect(c, depth, params, below, case_sensitive, refs)),
    }
}

/// Where one flattened-batch column comes from: every source repeats one
/// value per element of its row.
#[derive(Debug, Clone, Copy)]
enum Source {
    /// The input batch's column (a root column, or an enclosing flattened
    /// batch's column for a nested call), repeated per element.
    Root(usize),
    /// The list's child values.
    Element,
    /// Positions within each list.
    Index,
}

/// A count that does not fit its carrier.
fn too_many(n: usize) -> KrishivError {
    internal(&format!("a higher-order batch of {n} rows"))
}

/// A bound higher-order call.
#[derive(Debug)]
struct Call {
    name: String,
    array: Arc<dyn PhysExpr>,
    body: Arc<dyn PhysExpr>,
    sources: Vec<Source>,
    out: ArrowType,
    nullable: bool,
}

impl PhysExpr for Call {
    fn data_type(&self) -> &ArrowType {
        &self.out
    }

    fn nullable(&self) -> bool {
        self.nullable
    }

    fn eval(&self, batch: &RecordBatch) -> Result<Value> {
        let rows = batch.num_rows();
        let array = self.array.eval(batch)?.into_array(rows)?;
        if array.len() != rows {
            return Err(internal("a higher-order array of the wrong row count"));
        }
        let list = array
            .as_list_opt::<i32>()
            .ok_or_else(|| internal("a higher-order array that is no list"))?;
        let offsets = list.value_offsets();
        let at = |i: usize| {
            usize::try_from(offsets[i])
                .map_err(|_| internal("a higher-order list with a negative offset"))
        };
        let total = at(rows)?;
        if list.values().len() < total {
            return Err(internal("a higher-order list longer than its values"));
        }
        // The row each element belongs to: `offsets[r]..offsets[r+1]`.
        let mut rows_of: Vec<u32> = Vec::with_capacity(total);
        for r in 0..rows {
            let (start, stop) = (at(r)?, at(r.checked_add(1).ok_or_else(|| too_many(rows))?)?);
            let len = stop
                .checked_sub(start)
                .ok_or_else(|| internal("a higher-order list with decreasing offsets"))?;
            rows_of.extend(std::iter::repeat_n(
                u32::try_from(r).map_err(|_| too_many(rows))?,
                len,
            ));
        }
        let rows_of = UInt32Array::from(rows_of);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.sources.len());
        for source in &self.sources {
            columns.push(match source {
                // A root column, or an enclosing flattened batch's column
                // for a nested call: repeated per element.
                Source::Root(index) => {
                    let values = Arc::clone(batch.column(*index));
                    arrow_select::take::take(values.as_ref(), &rows_of, None)
                        .map_err(|e| internal(&e.to_string()))?
                }
                Source::Element => list.values().slice(0, total),
                Source::Index => {
                    let mut at_vals: Vec<i32> = Vec::with_capacity(total);
                    for r in 0..rows {
                        let next = r.checked_add(1).ok_or_else(|| too_many(rows))?;
                        let len = at(next)?.checked_sub(at(r)?).ok_or_else(|| {
                            internal("a higher-order list with decreasing offsets")
                        })?;
                        at_vals.extend(0..i32::try_from(len).map_err(|_| too_many(len))?);
                    }
                    Arc::new(Int32Array::from(at_vals))
                }
            });
        }
        let flat = if columns.is_empty() {
            // A body that reads nothing (a constant lambda) still needs
            // `total` rows to evaluate over; it ignores the column.
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("c0", ArrowType::Null, true)])),
                vec![Arc::new(arrow_array::NullArray::new(total)) as ArrayRef],
            )
            .map_err(|e| internal(&e.to_string()))?
        } else {
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
            .map_err(|e| internal(&e.to_string()))?
        };
        let body = self.body.eval(&flat)?.into_array(total)?;
        if body.len() != total {
            return Err(internal("a higher-order body of the wrong row count"));
        }
        let out = match self.name.as_str() {
            "transform" => {
                let ArrowType::List(field) = &self.out else {
                    return Err(internal("a higher-order call over a non-list"));
                };
                Arc::new(
                    list::transform_list(list, &body, field.is_nullable())
                        .map_err(|e| internal(&e.to_string()))?,
                ) as ArrayRef
            }
            "filter" | "exists" | "forall" => {
                let keep = body
                    .as_boolean_opt()
                    .ok_or_else(|| internal("a higher-order predicate that is no boolean array"))?;
                if self.name == "filter" {
                    Arc::new(list::filter_list(list, keep).map_err(|e| internal(&e.to_string()))?)
                        as ArrayRef
                } else if self.name == "exists" {
                    Arc::new(list::exists_list(list, keep).map_err(|e| internal(&e.to_string()))?)
                        as ArrayRef
                } else {
                    Arc::new(list::forall_list(list, keep).map_err(|e| internal(&e.to_string()))?)
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
