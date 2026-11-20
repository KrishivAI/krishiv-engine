// SPDX-License-Identifier: Apache-2.0
//! A grouped aggregate's output filter (`krishiv.exec.agg.outputFilter`):
//! a deterministic `HAVING` over the aggregate's values only, evaluated on
//! each output batch's values before the keys are decoded, so only the
//! groups that pass are decoded and handed on (TPC-H q18: 6,398 of 150M
//! groups). The plan's `Filter` above still runs, so a path that skips this
//! one (or a batch it lets through whole) gives the same rows; the
//! predicate sees the rows the `Filter` would see, so raises the same
//! errors (design `P3-A-aggregation` §3d).

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_schema::SchemaRef;
use arrow_select::filter::{filter, prep_null_mask_filter};

use super::HashAggregate;
use super::table::{AggCore, batch};
use crate::error::{Result, internal};
use crate::operators::Predicate;

/// The predicate and the value columns it reads.
#[derive(Debug)]
pub(crate) struct OutputFilter {
    /// Compiled over `schema`: the aggregate output columns `columns`.
    predicate: Arc<dyn Predicate>,
    /// Output column indexes (each past the keys), in `schema`'s order.
    columns: Vec<usize>,
    schema: SchemaRef,
}

impl OutputFilter {
    pub(crate) fn new(
        predicate: Arc<dyn Predicate>,
        columns: Vec<usize>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            predicate,
            columns,
            schema,
        }
    }

    /// The rows of `values` (the value columns, after `keys` key columns,
    /// of `rows` groups) that pass; `None` when every row passes.
    fn keep(&self, values: &[ArrayRef], keys: usize, rows: usize) -> Result<Option<BooleanArray>> {
        let cols = self
            .columns
            .iter()
            .map(|&c| {
                c.checked_sub(keys)
                    .and_then(|v| values.get(v))
                    .cloned()
                    .ok_or_else(|| internal(format!("output filter column {c} is not a value")))
            })
            .collect::<Result<Vec<_>>>()?;
        let mask = self.predicate.evaluate(&batch(&self.schema, cols, rows)?)?;
        let mask = if mask.null_count() > 0 {
            prep_null_mask_filter(&mask)
        } else {
            mask
        };
        Ok((mask.true_count() < rows).then_some(mask))
    }
}

/// The output batch of groups `ids` whose values are `values`: under an
/// output filter only the groups that pass, their keys decoded by `keys`.
pub(crate) fn output_rows(
    core: &AggCore,
    ids: &[u32],
    values: Vec<ArrayRef>,
    keys: impl FnOnce(&[u32]) -> Result<Vec<ArrayRef>>,
) -> Result<RecordBatch> {
    if let Some(f) = &core.output_filter
        && let Some(mask) = f.keep(&values, core.keys(), ids.len())?
    {
        let kept: Vec<u32> = ids
            .iter()
            .zip(mask.values().iter())
            .filter_map(|(&id, pass)| pass.then_some(id))
            .collect();
        let mut cols = keys(&kept)?;
        for v in &values {
            cols.push(filter(v, &mask).map_err(|e| internal(e.to_string()))?);
        }
        return batch(&core.output_schema, cols, kept.len());
    }
    let mut cols = keys(ids)?;
    cols.extend(values);
    batch(&core.output_schema, cols, ids.len())
}

impl HashAggregate {
    /// `agg` evaluating `filter` on its output's values first (see the
    /// module docs); unchanged when `agg` is already shared.
    pub(crate) fn with_output_filter(mut agg: Arc<Self>, filter: OutputFilter) -> Arc<Self> {
        if let Some(core) = Arc::get_mut(&mut agg).and_then(|a| Arc::get_mut(&mut a.core)) {
            core.output_filter = Some(filter);
        }
        agg
    }
}
