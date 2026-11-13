// SPDX-License-Identifier: Apache-2.0
//! The kernel's `EvaluationHandler` over Arrow 60 compute (design note P3-T021 section 3.3).
//!
//! A port of the semantics of the kernel's default evaluator (`delta_kernel` 0.30.1,
//! `src/engine/arrow_expression/` and the `parse_json`/`apply_schema` parts of
//! `src/engine/arrow_utils/`, Apache-2.0, copyright the Delta Lake project authors), which is
//! compiled only with the kernel's Arrow features (they pull `chrono-tz`, banned by ADR 0028).
//! It is not krishiv-exec's evaluator: the kernel's semantics are not Spark SQL's.
//!
//! The kernel uses it for log replay (`ParseJson` of `add.stats`, `MapToStruct` of
//! `add.partitionValues`, `COALESCE(stats, TO_JSON(stats_parsed))`), data skipping (a rewritten
//! predicate under `DISTINCT(output, false)`), transforms (`StructPatch`, `Struct`), writes
//! (`TO_JSON(stats)`), `create_many` for action rows, and in CDF on data batches, so it is
//! vectorised and allocates nothing per row. Node semantics: the modules `expr`, `predicate`,
//! `json` (`ParseJson`, `ToJson`) and `partition` (`MapToStruct`).
//!
//! An evaluator checks that its input batch has every field of its input schema, in order and
//! of a compatible top-level type (extra fields are allowed), and gives its output exactly the
//! declared type (`types::conform`): a struct output becomes the batch's columns, any other
//! output one nullable column named `output`.
//!
//! Differences from the kernel's evaluator, all toward the declared output: a widenable leaf
//! (an `Int32` where `long` is declared) is cast rather than left as it is, and `IN` over a
//! literal array gives one value per row rather than one value. Unsupported (the kernel's
//! `Unsupported`, never a wrong value): `Opaque` nodes (the kernel's Arrow adaptors). Zone names
//! in `MapToStruct` (a named reader zone, a partition value ending in one) resolve through
//! jiff's bundled zone database (see `partition`).

pub(crate) mod expr;
pub(crate) mod json;
pub(crate) mod partition;
pub(crate) mod predicate;
pub(crate) mod types;

use std::sync::Arc;

use arrow_array::builder::make_builder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, StructArray};
use arrow_schema::{ArrowError, DataType as ArrowType, Field, Schema};
use delta_kernel::expressions::{Expression, ExpressionRef, PredicateRef, Scalar};
use delta_kernel::schema::{DataType, SchemaRef, StructType};
use delta_kernel::{
    EngineData, EvaluationHandler, ExpressionEvaluator, KernelError, KernelResult,
    PredicateEvaluator,
};

use self::expr::{evaluate_expression, extract_column};
use self::predicate::evaluate_predicate;
use self::types::{check, conform};
use super::DeltaLimits;
use super::data::KrishivEngineData;
use super::schema::{append_scalar, to_arrow_schema, to_arrow_type};

/// What every evaluation needs: the parse limits `ParseJson` applies.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ctx {
    pub(crate) limits: DeltaLimits,
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "used as `map_err(arrow_err)`"
)]
pub(crate) fn arrow_err(e: ArrowError) -> KernelError {
    KernelError::generic(e.to_string())
}

/// The query's cancellation as the evaluation handler sees it: none for a standalone handler.
#[derive(Debug, Clone, Default)]
struct Cancel(Option<crate::engine::Cancel>);

impl Cancel {
    /// The kernel's `Cancelled` once the query is cancelled, as the other handlers return.
    fn check(&self) -> KernelResult<()> {
        match &self.0 {
            Some(c) if c.is_cancelled() => Err(KernelError::Cancelled),
            _ => Ok(()),
        }
    }
}

/// The evaluation handler of a [`super::DeltaEngine`]. Every call, and every evaluator's
/// `evaluate`, checks the query's cancellation first.
#[derive(Debug, Clone)]
pub struct KrishivEvaluationHandler {
    ctx: Ctx,
    cancel: Cancel,
}

impl KrishivEvaluationHandler {
    /// A handler applying `limits` to `ParseJson`, with no query to cancel it.
    #[must_use]
    pub fn new(limits: DeltaLimits) -> Self {
        Self {
            ctx: Ctx { limits },
            cancel: Cancel::default(),
        }
    }

    /// A handler of the query whose cancellation is `cancel`.
    pub(crate) fn with_cancel(limits: DeltaLimits, cancel: crate::engine::Cancel) -> Self {
        Self {
            ctx: Ctx { limits },
            cancel: Cancel(Some(cancel)),
        }
    }
}

impl EvaluationHandler for KrishivEvaluationHandler {
    fn new_expression_evaluator(
        &self,
        input_schema: SchemaRef,
        expression: ExpressionRef,
        output_type: DataType,
    ) -> KernelResult<Arc<dyn ExpressionEvaluator>> {
        self.cancel.check()?;
        Ok(Arc::new(KrishivExpressionEvaluator {
            ctx: self.ctx,
            cancel: self.cancel.clone(),
            input_schema,
            expression,
            output_type,
        }))
    }

    fn new_predicate_evaluator(
        &self,
        input_schema: SchemaRef,
        predicate: PredicateRef,
    ) -> KernelResult<Arc<dyn PredicateEvaluator>> {
        self.cancel.check()?;
        Ok(Arc::new(KrishivPredicateEvaluator {
            ctx: self.ctx,
            cancel: self.cancel.clone(),
            input_schema,
            predicate,
        }))
    }

    fn create_many(
        &self,
        schema: SchemaRef,
        rows: Vec<Vec<Scalar>>,
    ) -> KernelResult<Box<dyn EngineData>> {
        self.cancel.check()?;
        Ok(Box::new(KrishivEngineData::new(create_many(
            &schema, &rows,
        )?)))
    }
}

/// `t` with every struct field, list element and map value nullable: the builders' type, so a
/// misplaced null is an error from [`conform`] rather than a panic in a builder.
fn all_nullable(t: &ArrowType) -> ArrowType {
    let loose = |f: &Field| {
        Arc::new(
            Field::new(f.name(), all_nullable(f.data_type()), true)
                .with_metadata(f.metadata().clone()),
        )
    };
    match t {
        ArrowType::Struct(fields) => ArrowType::Struct(fields.iter().map(|f| loose(f)).collect()),
        ArrowType::List(f) => ArrowType::List(loose(f)),
        ArrowType::Map(entries, sorted) => {
            let ArrowType::Struct(kv) = entries.data_type() else {
                return t.clone();
            };
            let kv: Vec<_> = kv
                .iter()
                .enumerate()
                .map(|(i, f)| if i == 0 { Arc::clone(f) } else { loose(f) })
                .collect();
            ArrowType::Map(
                Arc::new(Field::new(
                    entries.name(),
                    ArrowType::Struct(kv.into()),
                    false,
                )),
                *sorted,
            )
        }
        other => other.clone(),
    }
}

/// One batch of `rows`, one scalar per top-level field of `schema` each.
///
/// # Errors
/// A row of the wrong width, a non-null scalar whose type is not the declared one (a decimal
/// of another precision or scale is refused, never rescaled), or a null in a non-nullable
/// field.
pub(crate) fn create_many(schema: &StructType, rows: &[Vec<Scalar>]) -> KernelResult<RecordBatch> {
    let arrow = Arc::new(to_arrow_schema(schema)?);
    let width = schema.num_fields();
    for (i, row) in rows.iter().enumerate() {
        if row.len() != width {
            return Err(KernelError::generic(format!(
                "Row {i} has {} scalars but schema has {width} fields",
                row.len()
            )));
        }
    }
    let mut columns = Vec::with_capacity(width);
    for (col, (field, arrow_field)) in schema.fields().zip(arrow.fields()).enumerate() {
        let mut b = make_builder(&all_nullable(arrow_field.data_type()), rows.len());
        for (i, row) in rows.iter().enumerate() {
            let value = &row[col];
            let fail = |e: &dyn std::fmt::Display| {
                KernelError::generic(format!(
                    "Row {i}, field '{}' (expected type {}, got {}): {e}",
                    field.name(),
                    field.data_type(),
                    value.data_type()
                ))
            };
            // A builder takes the bits of a value of another decimal scale or zone label
            // without complaint, so the type is checked first (the kernel's `create_many`
            // does not check; it never rescales either).
            if !value.is_null() && !same_type(&value.data_type(), field.data_type()) {
                return Err(fail(&"the scalar's type does not match the declared type"));
            }
            append_scalar(value, b.as_mut(), 1).map_err(|e| fail(&e))?;
        }
        columns.push(conform(&b.finish(), arrow_field.data_type())?);
    }
    RecordBatch::try_new_with_options(
        arrow,
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows.len())),
    )
    .map_err(arrow_err)
}

/// Whether a scalar of type `got` fills a column declared `want`: the same types, struct
/// fields by position, nullability, field names and metadata aside (the builders are
/// positional and [`conform`] applies the declared names).
fn same_type(got: &DataType, want: &DataType) -> bool {
    match (got, want) {
        (DataType::Primitive(a), DataType::Primitive(b)) => a == b,
        (DataType::Array(a), DataType::Array(b)) => same_type(a.element_type(), b.element_type()),
        (DataType::Map(a), DataType::Map(b)) => {
            same_type(a.key_type(), b.key_type()) && same_type(a.value_type(), b.value_type())
        }
        (DataType::Struct(a), DataType::Struct(b))
        | (DataType::Variant(a), DataType::Variant(b)) => {
            a.num_fields() == b.num_fields()
                && a.fields()
                    .zip(b.fields())
                    .all(|(x, y)| same_type(x.data_type(), y.data_type()))
        }
        _ => false,
    }
}

/// Fails unless `batch` has each field of `expected` in order (extra fields allowed), each of
/// a compatible top-level type.
fn check_input(expected: &StructType, batch: &RecordBatch) -> KernelResult<()> {
    let fields = batch.schema_ref().fields();
    let mut rest = fields.iter();
    for want in expected.fields() {
        let Some(got) = rest.find(|f| f.name() == want.name()) else {
            let how = if fields.iter().any(|f| f.name() == want.name()) {
                "out of order"
            } else {
                "missing"
            };
            return Err(KernelError::schema(format!(
                "Expected schema field '{}' is {how} in data schema fields {:?}",
                want.name(),
                fields.iter().map(|f| f.name()).collect::<Vec<_>>()
            )));
        };
        let compatible = match (want.data_type(), got.data_type()) {
            (DataType::Primitive(_), _) => check(want.data_type(), got.data_type()).is_ok(),
            (DataType::Struct(_) | DataType::Variant(_), ArrowType::Struct(_))
            | (
                DataType::Array(_),
                ArrowType::List(_)
                | ArrowType::LargeList(_)
                | ArrowType::ListView(_)
                | ArrowType::LargeListView(_)
                | ArrowType::FixedSizeList(..),
            )
            | (DataType::Map(_), ArrowType::Map(..)) => true,
            _ => false,
        };
        if !compatible {
            return Err(KernelError::schema(format!(
                "Expected schema type for '{}' does not match the data schema type: {:?} != {:?}",
                want.name(),
                want.data_type(),
                got.data_type()
            )));
        }
    }
    Ok(())
}

/// The struct column `array` as a batch (its own nulls dropped: a batch has no row nulls).
fn struct_to_batch(array: &ArrayRef) -> KernelResult<RecordBatch> {
    let s: StructArray = array
        .as_struct_opt()
        .ok_or_else(|| KernelError::generic("a struct output is not a struct array"))?
        .clone();
    let rows = s.len();
    let (fields, columns, _) = s.into_parts();
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .map_err(arrow_err)
}

#[derive(Debug)]
struct KrishivExpressionEvaluator {
    ctx: Ctx,
    cancel: Cancel,
    input_schema: SchemaRef,
    expression: ExpressionRef,
    output_type: DataType,
}

impl ExpressionEvaluator for KrishivExpressionEvaluator {
    fn evaluate(&self, batch: &dyn EngineData) -> KernelResult<Box<dyn EngineData>> {
        self.cancel.check()?;
        let batch = KrishivEngineData::batch_of(batch)?;
        check_input(&self.input_schema, batch)?;
        let out = evaluate(&self.ctx, &self.expression, &self.output_type, batch)?;
        Ok(Box::new(KrishivEngineData::new(out)))
    }
}

/// Evaluates `expression` over `batch` as `output_type` (see the module docs).
pub(crate) fn evaluate(
    ctx: &Ctx,
    expression: &Expression,
    output_type: &DataType,
    batch: &RecordBatch,
) -> KernelResult<RecordBatch> {
    let target = to_arrow_type(output_type)?;
    match (expression, output_type) {
        (Expression::StructPatch(patch), DataType::Struct(_)) if patch.is_empty() => {
            // Only the output schema changes (column mapping): no evaluation.
            let array: ArrayRef = match patch.input_path() {
                None => Arc::new(StructArray::from(batch.clone())),
                Some(path) => extract_column(batch, path.path())?,
            };
            struct_to_batch(&conform(&array, &target)?)
        }
        (_, DataType::Struct(_)) => {
            let array = evaluate_expression(ctx, expression, batch, Some(output_type))?;
            struct_to_batch(&conform(&array, &target)?)
        }
        _ => {
            let array = evaluate_expression(ctx, expression, batch, Some(output_type))?;
            let array = conform(&array, &target)?;
            let schema = Schema::new(vec![Field::new("output", target, true)]);
            RecordBatch::try_new_with_options(
                Arc::new(schema),
                vec![array],
                &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )
            .map_err(arrow_err)
        }
    }
}

#[derive(Debug)]
struct KrishivPredicateEvaluator {
    ctx: Ctx,
    cancel: Cancel,
    input_schema: SchemaRef,
    predicate: PredicateRef,
}

impl PredicateEvaluator for KrishivPredicateEvaluator {
    fn evaluate(&self, batch: &dyn EngineData) -> KernelResult<Box<dyn EngineData>> {
        self.cancel.check()?;
        let batch = KrishivEngineData::batch_of(batch)?;
        check_input(&self.input_schema, batch)?;
        let result = evaluate_predicate(&self.ctx, &self.predicate, batch, false)?;
        let schema = Schema::new(vec![Field::new("output", ArrowType::Boolean, true)]);
        let out =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(result)]).map_err(arrow_err)?;
        Ok(Box::new(KrishivEngineData::new(out)))
    }
}

#[cfg(test)]
mod tests;
