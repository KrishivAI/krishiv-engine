// SPDX-License-Identifier: Apache-2.0
//! Generators: a row per element of an array or map column (Spark's
//! `Generate` with `Explode`, `PosExplode` and `Inline`).

use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, UInt32Array};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use arrow_select::take::{take, take_record_batch};
use krishiv_plan::GeneratorKind;
use krishiv_types::batch::{Batch, Weights};

use super::expr::ScalarExpr;
use super::local_mut;
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{LocalState, Operator, OperatorOutput};

/// For each input row, the rows of the argument's elements it generates: a
/// row keeps its weight in every copy. With `outer`, a row whose argument
/// is NULL or empty is kept once with NULL generated columns; without, it
/// is dropped. One output batch per input batch (a row's elements are
/// never split), so a very long array makes one long batch.
#[derive(Debug)]
pub struct Generate {
    schema: SchemaRef,
    kind: GeneratorKind,
    outer: bool,
    arg: Arc<dyn ScalarExpr>,
}

impl Generate {
    /// Generates from `arg` over batches of `input`: the output is the
    /// input's columns followed by the generated ones, typed after the
    /// argument.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the argument's type is not one the generator
    /// takes (the analyzer checked it).
    pub fn try_new(
        input: &SchemaRef,
        kind: GeneratorKind,
        outer: bool,
        arg: Arc<dyn ScalarExpr>,
    ) -> Result<Arc<Self>> {
        let mut fields: Vec<Arc<Field>> = input.fields().iter().cloned().collect();
        fields.extend(generated_fields(kind, outer, &arg.data_type())?);
        let schema = Arc::new(Schema::new(fields));
        Ok(Arc::new(Self {
            schema,
            kind,
            outer,
            arg,
        }))
    }
}

/// The generated columns for an argument of Arrow type `ty`.
fn generated_fields(kind: GeneratorKind, outer: bool, ty: &DataType) -> Result<Vec<Arc<Field>>> {
    let mut out = Vec::new();
    if kind == GeneratorKind::PosExplode {
        out.push(Arc::new(Field::new("pos", DataType::Int32, outer)));
    }
    let element = |f: &Field| {
        Arc::new(Field::new(
            f.name().as_str(),
            f.data_type().clone(),
            f.is_nullable() || outer,
        ))
    };
    match (kind, ty) {
        (GeneratorKind::Explode | GeneratorKind::PosExplode, DataType::List(f)) => {
            out.push(Arc::new(Field::new(
                "col",
                f.data_type().clone(),
                f.is_nullable() || outer,
            )));
        }
        (GeneratorKind::Explode | GeneratorKind::PosExplode, DataType::Map(entries, _)) => {
            let DataType::Struct(kv) = entries.data_type() else {
                return Err(internal("a map whose entries are not a struct".to_owned()));
            };
            for (i, f) in kv.iter().enumerate() {
                let name = if i == 0 { "key" } else { "value" };
                out.push(Arc::new(Field::new(
                    name,
                    f.data_type().clone(),
                    // Keys are never null, except in the NULL row of an outer
                    // generator.
                    f.is_nullable() || outer,
                )));
            }
        }
        (GeneratorKind::Inline, DataType::List(f)) => {
            let DataType::Struct(fields) = f.data_type() else {
                return Err(internal(
                    "inline of an array that is not of structs".to_owned(),
                ));
            };
            out.extend(fields.iter().map(|f| element(f)));
        }
        _ => {
            return Err(internal(format!("a {kind:?} generator over a {ty} column")));
        }
    }
    Ok(out)
}

/// The driver's partition: `arg` evaluates stateful functions in it.
struct Partition(u32);

/// The elements of an argument column: its offsets (the element range of
/// each row, `None` for a NULL row) and the arrays the elements are taken
/// from.
struct Elements {
    ranges: Vec<Option<(usize, usize)>>,
    /// The columns the elements are read from, in output order, before any
    /// position.
    sources: Vec<ArrayRef>,
}

impl Elements {
    fn of(kind: GeneratorKind, arg: &ArrayRef) -> Result<Self> {
        let offsets_ranges =
            |offsets: &OffsetBuffer<i32>, nulls: Option<&arrow_buffer::NullBuffer>| {
                (0..offsets.len().saturating_sub(1))
                    .map(|i| {
                        if nulls.is_some_and(|n| n.is_null(i)) {
                            None
                        } else {
                            let start = usize::try_from(offsets[i]).ok()?;
                            let end = usize::try_from(offsets[i + 1]).ok()?;
                            Some((start, end))
                        }
                    })
                    .collect::<Vec<_>>()
            };
        if let Some(list) = arg.as_list_opt::<i32>() {
            let ranges = offsets_ranges(list.offsets(), list.nulls());
            let values = Arc::clone(list.values());
            let sources = if kind == GeneratorKind::Inline {
                let s = values.as_struct_opt().ok_or_else(|| {
                    internal("inline of an array that is not of structs".to_owned())
                })?;
                s.columns().to_vec()
            } else {
                vec![values]
            };
            return Ok(Self { ranges, sources });
        }
        if let Some(map) = arg.as_map_opt() {
            if kind == GeneratorKind::Inline {
                return Err(internal("inline of a map".to_owned()));
            }
            let ranges = offsets_ranges(map.offsets(), map.nulls());
            let entries = map.entries();
            return Ok(Self {
                ranges,
                sources: vec![Arc::clone(entries.column(0)), Arc::clone(entries.column(1))],
            });
        }
        Err(internal(format!(
            "a generator over a {} column",
            arg.data_type()
        )))
    }
}

impl Operator for Generate {
    fn name(&self) -> &'static str {
        "Generate"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn local_state(&self, ctx: &TaskContext) -> Result<LocalState> {
        Ok(Box::new(Partition(ctx.partition())))
    }

    fn execute(&self, input: &Batch, local: &mut LocalState) -> Result<OperatorOutput> {
        if input.num_rows() == 0 {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let partition = local_mut::<Partition>(local, "Generate")?.0;
        let arg = self.arg.evaluate_in(input.data(), partition)?;
        let elements = Elements::of(self.kind, &arg)?;
        // The input row and the element each output row reads (`None`: the
        // NULL row of an outer generator), and its position.
        let mut rows: Vec<u32> = Vec::new();
        let mut taken: Vec<Option<u32>> = Vec::new();
        let mut positions: Vec<Option<i32>> = Vec::new();
        for (row, range) in elements.ranges.iter().enumerate() {
            let row32 =
                u32::try_from(row).map_err(|_| internal("a batch over 2^32 rows".to_owned()))?;
            let (start, end) = range.unwrap_or((0, 0));
            if end > start {
                for (pos, element) in (start..end).enumerate() {
                    rows.push(row32);
                    taken
                        .push(Some(u32::try_from(element).map_err(|_| {
                            internal("an element index over 2^32".to_owned())
                        })?));
                    positions.push(Some(
                        i32::try_from(pos)
                            .map_err(|_| internal("a position over 2^31".to_owned()))?,
                    ));
                }
            } else if self.outer {
                rows.push(row32);
                taken.push(None);
                positions.push(None);
            }
        }
        if rows.is_empty() {
            return Ok(OperatorOutput::NeedMoreInput(None));
        }
        let arrow = |e: arrow_schema::ArrowError| internal(e.to_string());
        let indices = UInt32Array::from(rows);
        // A relation without columns (`FROM explode(...)`) has rows only.
        let mut columns: Vec<ArrayRef> = if input.data().num_columns() == 0 {
            Vec::new()
        } else {
            take_record_batch(input.data(), &indices)
                .map_err(arrow)?
                .columns()
                .to_vec()
        };
        if matches!(self.kind, GeneratorKind::PosExplode) {
            columns.push(Arc::new(Int32Array::from(positions)));
        }
        let element_indices = UInt32Array::from(taken);
        for source in &elements.sources {
            columns.push(take(source.as_ref(), &element_indices, None).map_err(arrow)?);
        }
        let data = RecordBatch::try_new(Arc::clone(&self.schema), columns).map_err(arrow)?;
        let weights = match input.weights() {
            Weights::Const(w) => Weights::Const(*w),
            Weights::Column(c) => Weights::Column(
                take(c, &indices, None)
                    .map_err(arrow)?
                    .as_primitive()
                    .clone(),
            ),
        };
        Ok(OperatorOutput::NeedMoreInput(Some(Batch::with_weights(
            data, weights,
        )?)))
    }
}
