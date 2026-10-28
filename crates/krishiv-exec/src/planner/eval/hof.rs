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
use arrow_array::{Array, ArrayRef, Int32Array, ListArray, RecordBatch, UInt32Array};
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
    collect(body, params, &mut Vec::new(), &mut refs);
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
/// lambda variables (each level carries what the levels below need), and
/// this call's element and index variables.
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

/// Collects `refs` from `body` against `params`, the current call's
/// parameters with their stable identities (a nested call collects its own
/// variables when it compiles: here its parameters only shadow, through
/// `below`). A variable of an enclosing call is an outer reference here,
/// carried through this level's flattened batch. Matching is by
/// [`AttrId`], never by name, so case sensitivity cannot change it.
fn collect(body: &Arc<Expr>, params: &[BoundColumn], below: &mut Vec<AttrId>, refs: &mut Refs) {
    match &body.node {
        ExprNode::Lambda {
            params: inner,
            body: nested,
        } => {
            // The nested call collects its own variables when it compiles:
            // this pass only gathers what the current level carries, so its
            // parameters stay in scope and the nested ones shadow them.
            below.extend(inner.iter().map(|p| p.id));
            collect(nested, params, below, refs);
            below.truncate(below.len() - inner.len());
        }
        ExprNode::Resolved(ResolvedExpr::LambdaVar(c)) => {
            if below.contains(&c.id) {
                // A variable a nested call binds is that call's own: its
                // flattening carries it, never this one.
            } else if params.first().is_some_and(|p| p.id == c.id) {
                // This call's element variable, used directly or from a
                // nested lambda: the flattened batch carries one element
                // column per use, so both read it here.
                if !refs.lambda.contains(&c.id) {
                    refs.lambda.push(c.id);
                }
            } else if params.iter().skip(1).any(|p| p.id == c.id) {
                if refs.index.is_none() {
                    refs.index = Some(c.id);
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
        _ => body.for_each_child(&mut |c| collect(c, params, below, refs)),
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

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        let array = self.array.eval(batch, partition)?.into_array(rows)?;
        if array.len() != rows {
            return Err(internal("a higher-order array of the wrong row count"));
        }
        let list = array
            .as_list_opt::<i32>()
            .ok_or_else(|| internal("a higher-order array that is no list"))?;
        // Rebase a sliced list to zero-based offsets once: a slice keeps
        // the absolute offsets, so `offsets[0]` may be non-zero while the
        // kernels see zero-based lists.
        let at = |offsets: &[i32], i: usize| {
            usize::try_from(offsets[i])
                .map_err(|_| internal("a higher-order list with a negative offset"))
        };
        let offsets = list.value_offsets();
        let base = at(offsets, 0)?;
        let end = at(offsets, rows)?;
        let total = end
            .checked_sub(base)
            .ok_or_else(|| internal("a higher-order list with decreasing offsets"))?;
        let rebased = {
            let mut norm = Vec::with_capacity(rows + 1);
            for i in 0..=rows {
                let v = at(offsets, i)?
                    .checked_sub(base)
                    .ok_or_else(|| internal("a higher-order list with decreasing offsets"))?;
                norm.push(
                    i32::try_from(v)
                        .map_err(|_| internal("a higher-order list with a negative offset"))?,
                );
            }
            ListArray::new(
                Arc::clone(list.value_field()),
                arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(norm)),
                list.values().slice(base, total),
                list.nulls().cloned(),
            )
        };
        let list = &rebased;
        let offsets = list.value_offsets();
        let at = |i: usize| {
            usize::try_from(offsets[i])
                .map_err(|_| internal("a higher-order list with a negative offset"))
        };
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
                    // Positions within each row's run of `rows_of` (built
                    // above: one entry per element, in row order), so each
                    // new row restarts the count. The offsets were validated
                    // non-decreasing there, and a run holds at most `total`
                    // entries (`total` fits `i32`, as it is an offset), so
                    // the count cannot overflow.
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
            kernels: KernelCtx::new(),
            query_start_micros: 0,
            user: Arc::from("test"),
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
        c.sources = vec![super::Source::Element];
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
}
