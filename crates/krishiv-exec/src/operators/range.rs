// SPDX-License-Identifier: Apache-2.0
//! The `range` source: `range(start, end, step, numSlices)` rows.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::future::BoxFuture;
use krishiv_types::batch::Batch;

use super::global_ref;
use crate::MorselCursor;
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState};

/// `range(start, end, step, numSlices)`: the `id BIGINT NOT NULL` values.
#[derive(Debug, Clone)]
pub(crate) struct Range {
    start: i64,
    end: i64,
    step: i64,
    num_slices: Option<u32>,
}

impl Range {
    /// A new range source. `step` must not be zero (the analyzer rejects
    /// it); `num_slices` is the written partition count, if any.
    #[must_use]
    pub(crate) fn new(start: i64, end: i64, step: i64, num_slices: Option<u32>) -> Self {
        Self {
            start,
            end,
            step,
            num_slices,
        }
    }
}

struct RangeGlobal {
    start: i64,
    end: i64,
    step: i64,
    partitions: u32,
    cursor: MorselCursor,
    batch_size: usize,
    returned: AtomicU64,
}

struct RangeLocal {
    /// `(value, exclusive end)` of the current slice, or `None` to claim.
    current: Option<(i128, i128)>,
    step: i64,
    claimed: u64,
}

fn range_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]))
}

impl crate::operator::Source for Range {
    fn name(&self) -> &'static str {
        "Range"
    }

    fn schema(&self) -> SchemaRef {
        range_schema()
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let config = ctx.query().config();
        let wanted = match self.num_slices {
            Some(n) => n.max(1),
            None => u32::try_from(ctx.executor().parallelism().max(1)).unwrap_or(u32::MAX),
        };
        // Slices beyond the row count are empty: they only cost the
        // morsel cursor (a seeded one allocates a permutation of them).
        let rows =
            krishiv_plan::analyzer::table_functions::range_count(self.start, self.end, self.step);
        let partitions = u32::try_from(rows.clamp(1, i128::from(wanted))).unwrap_or(wanted);
        Ok(Arc::new(RangeGlobal {
            start: self.start,
            end: self.end,
            step: self.step,
            partitions,
            cursor: MorselCursor::new(partitions as usize, config.seed),
            batch_size: config.batch_size.max(1),
            returned: AtomicU64::new(0),
        }))
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        global_ref::<RangeGlobal>(global, "Range").map_or(1, |g| (g.partitions as usize).max(1))
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(RangeLocal {
            current: None,
            step: self.step,
            claimed: 0,
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(std::future::ready(self.next_batch(local, global)))
    }

    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        local.downcast_ref::<RangeLocal>().map_or(0, |l| l.claimed)
    }
}

impl Range {
    fn next_batch(&self, local: &mut LocalState, global: &GlobalState) -> Result<Option<Batch>> {
        let g = global_ref::<RangeGlobal>(global, "Range")?;
        let local = local
            .downcast_mut::<RangeLocal>()
            .ok_or_else(|| internal("wrong local state for Range".to_owned()))?;
        loop {
            match local.current {
                None => {
                    let Some(index) = g.cursor.claim() else {
                        return Ok(None);
                    };
                    local.claimed += 1;
                    match krishiv_plan::analyzer::table_functions::range_slice(
                        g.start,
                        g.end,
                        g.step,
                        u32::try_from(index).map_err(|_| {
                            crate::error::internal("a range partition past u32".to_owned())
                        })?,
                        g.partitions,
                    ) {
                        None => continue,
                        Some((first, exclusive)) => {
                            local.current = Some((i128::from(first), i128::from(exclusive)));
                        }
                    }
                }
                Some((cur, hi)) => {
                    let step = i128::from(local.step);
                    debug_assert!(step != 0);
                    let empty = if step > 0 { cur >= hi } else { cur <= hi };
                    if empty {
                        local.current = None;
                        continue;
                    }
                    let batch_size = g.batch_size;
                    let mut values = Vec::with_capacity(batch_size);
                    let mut v = cur;
                    while values.len() < batch_size && (if step > 0 { v < hi } else { v > hi }) {
                        let value = i64::try_from(v)
                            .map_err(|_| internal("a range value outside i64".to_owned()))?;
                        values.push(value);
                        v += step;
                    }
                    local.current = Some((v, hi));
                    g.returned.fetch_add(values.len() as u64, Ordering::Relaxed);
                    let array = Int64Array::from(values);
                    let batch = RecordBatch::try_new(range_schema(), vec![Arc::new(array) as _])
                        .map_err(|e| internal(e.to_string()))?;
                    return Ok(Some(Batch::new(batch)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Source as _;
    use futures::FutureExt as _;
    use krishiv_memory::{MemoryConfig, MemoryPool as _, SpillOwner, TreePool};
    use std::sync::Arc;

    use crate::{ExecConfig, ExecutorConfig, OperatorId, QueryContext, QueryId, WorkloadClass};

    fn ctx_with(config: ExecConfig) -> (Arc<TreePool>, TaskContext) {
        let root = TreePool::new(&MemoryConfig::default());
        let q = QueryContext::new(
            QueryId(1),
            WorkloadClass::Batch,
            config,
            root.child("q", None),
        );
        let exec = crate::executor::TokioExecutor::new(&ExecutorConfig::default())
            .unwrap_or_else(|e| unreachable!("{e}"));
        let ctx = TaskContext::new(q, exec, SpillOwner::new(), OperatorId(0), "op");
        (root, ctx)
    }

    fn drain(source: &Range, ctx: &TaskContext) -> Vec<Batch> {
        let g = source
            .global_state(ctx)
            .unwrap_or_else(|e| unreachable!("{e}"));
        let mut local = source
            .local_state(ctx, &g)
            .unwrap_or_else(|e| unreachable!("{e}"));
        let mut out = Vec::new();
        while let Some(b) = source
            .get_data(&mut local, &g)
            .now_or_never()
            .unwrap_or_else(|| unreachable!("range does not wait"))
            .unwrap_or_else(|e| unreachable!("{e}"))
        {
            // No empty batches, weight 1.
            assert!(b.num_rows() > 0);
            assert_eq!(b.weights(), &krishiv_types::batch::Weights::Const(1));
            out.push(b);
        }
        out
    }

    fn values(batches: &[Batch]) -> Vec<i64> {
        use arrow_array::cast::AsArray as _;
        use arrow_array::types::Int64Type;
        let mut out = Vec::new();
        for b in batches {
            let col = b
                .data()
                .column(0)
                .as_primitive::<Int64Type>()
                .values()
                .to_vec();
            out.extend(col);
        }
        out
    }

    #[test]
    fn range_emits_batches_of_configured_size() {
        let (_root, ctx) = ctx_with(ExecConfig {
            batch_size: 4,
            ..ExecConfig::default()
        });
        let source = Range::new(0, 10, 1, Some(1));
        let out = drain(&source, &ctx);
        // 10 rows in batches of 4: 4 + 4 + 2.
        assert_eq!(
            out.iter().map(Batch::num_rows).collect::<Vec<_>>(),
            vec![4, 4, 2]
        );
        assert_eq!(values(&out), (0..10).collect::<Vec<_>>());
        assert_eq!(source.schema().fields().len(), 1);
    }

    #[test]
    fn range_empty_and_negative_steps() {
        let (_root, ctx) = ctx_with(ExecConfig::default());
        // `range(5, 1)` has no rows.
        let out = drain(&Range::new(5, 1, 1, Some(1)), &ctx);
        assert!(out.is_empty());
        // A negative step counts down.
        let out = drain(&Range::new(5, 1, -2, Some(1)), &ctx);
        assert_eq!(values(&out), vec![5, 3]);
        // `range(1, 10, 3, 2)` is 1, 4, 7 across two slices.
        let out = drain(&Range::new(1, 10, 3, Some(2)), &ctx);
        let mut got = values(&out);
        got.sort_unstable();
        assert_eq!(got, vec![1, 4, 7]);
    }

    #[test]
    fn range_partitioning_uses_num_slices_or_parallelism() {
        let (_root, ctx) = ctx_with(ExecConfig::default());
        // `numSlices` is the partition count.
        let source = Range::new(0, 100, 1, Some(4));
        let g = source
            .global_state(&ctx)
            .unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(source.max_parallelism(&g), 4);
        // Without it the task's parallelism decides; the fragment never
        // stores a count it did not get.
        let source = Range::new(0, 10, 1, None);
        assert_eq!(source.num_slices, None);
        let g = source
            .global_state(&ctx)
            .unwrap_or_else(|e| unreachable!("{e}"));
        assert!(source.max_parallelism(&g) >= 1);
    }

    /// More slices than rows are clamped to the row count, so a huge
    /// `numSlices` costs nothing (a seeded cursor would otherwise allocate
    /// a permutation of every slice) and still returns every row once.
    #[test]
    fn range_clamps_slices_to_the_row_count() {
        let (_root, ctx) = ctx_with(ExecConfig {
            seed: Some(7),
            ..ExecConfig::default()
        });
        let source = Range::new(0, 5, 1, Some(i32::MAX.cast_unsigned()));
        let g = source
            .global_state(&ctx)
            .unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(source.max_parallelism(&g), 5);
        let mut got = values(&drain(&source, &ctx));
        got.sort_unstable();
        assert_eq!(got, vec![0, 1, 2, 3, 4]);
        // An empty range keeps one (empty) slice.
        let source = Range::new(5, 5, 1, Some(1_000_000));
        let g = source
            .global_state(&ctx)
            .unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(source.max_parallelism(&g), 1);
        assert!(drain(&source, &ctx).is_empty());
    }

    #[test]
    fn range_long_max_boundary_returns_one_row() {
        let (_root, ctx) = ctx_with(ExecConfig::default());
        let out = drain(&Range::new(i64::MAX - 1, i64::MAX, 1, Some(1)), &ctx);
        assert_eq!(values(&out), vec![i64::MAX - 1]);
    }
}
