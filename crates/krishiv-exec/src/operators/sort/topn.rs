// SPDX-License-Identifier: Apache-2.0
//! `ORDER BY … LIMIT k [OFFSET m]` as a bounded breaker.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_row::Rows;
use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivUnsupportedFeature;
use krishiv_memory::Reservation;
use krishiv_types::batch::Batch;

use super::finalized;
use super::keys::{KeyEncoder, SortKey};
use super::merge::interleave_batches;
use crate::context::TaskContext;
use crate::error::{Result, internal};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::operators::{global_ref, local_mut};
use crate::sync::{Mutex, lock};

/// The rows kept so far, sorted, with their key rows.
#[derive(Debug)]
struct Top {
    batch: Batch,
    rows: Rows,
    reservations: Vec<Reservation>,
}

impl Top {
    fn len(&self) -> usize {
        self.batch.num_rows()
    }

    fn bytes(batch: &Batch, rows: &Rows) -> usize {
        batch.data().get_array_memory_size() + rows.size()
    }
}

/// Keeps the first `offset + limit` rows of its input in key order, so it
/// never holds more than that many rows (plus the batch being sunk), and
/// emits rows `offset..offset + limit` of them. The planner uses it for
/// `ORDER BY … LIMIT` up to [`TopN::MAX_ROWS`] rows and a [`Sort`](super::Sort)
/// with a `Limit` above.
///
/// **Per batch** the sink compares every row with the current `k`-th row
/// and merges only the rows that beat it, so most batches of a long input
/// cost one comparison per row; a batch with candidates costs
/// `O(k + c log c)`. Ties keep the earlier row, so within one driver the
/// output is the stable sort's prefix. Drivers merge their tops in
/// `combine`.
///
/// **Weights:** defined for insert-only input; weighted (change stream)
/// input is rejected with `KRISHIV_UNSUPPORTED_FEATURE` (incremental top-N
/// over a change stream needs retraction-aware state).
#[derive(Debug)]
pub struct TopN {
    schema: SchemaRef,
    encoder: Arc<KeyEncoder>,
    offset: usize,
    keep: usize,
    max_retained: AtomicU64,
}

impl TopN {
    /// The largest `offset + limit` a `TopN` keeps; beyond it, plan a sort
    /// and a limit.
    pub const MAX_ROWS: usize = 1 << 16;

    /// `ORDER BY keys LIMIT limit OFFSET offset` over batches of `schema`.
    ///
    /// # Errors
    /// As [`Sort::new`](super::Sort::new); `INTERNAL_ERROR` if
    /// `offset + limit` exceeds [`Self::MAX_ROWS`].
    pub fn new(schema: SchemaRef, keys: &[SortKey], offset: usize, limit: usize) -> Result<Self> {
        let keep = offset
            .checked_add(limit)
            .filter(|k| *k <= Self::MAX_ROWS)
            .ok_or_else(|| {
                internal(format!(
                    "TopN keeps at most {} rows; plan a Sort and a Limit for OFFSET {offset} LIMIT {limit}",
                    Self::MAX_ROWS
                ))
            })?;
        Ok(Self {
            encoder: KeyEncoder::new(&schema, keys)?,
            schema,
            offset,
            keep,
            max_retained: AtomicU64::new(0),
        })
    }

    /// The most rows one driver's (or the combined) state held after a
    /// batch: at most `offset + limit`.
    pub fn max_retained_rows(&self) -> u64 {
        self.max_retained.load(Ordering::Relaxed)
    }

    /// The first `keep` rows of `a` merged with rows `b_idx` (sorted) of
    /// `b`; ties take `a` first.
    fn merge(
        &self,
        a: Option<&Top>,
        b: &Batch,
        b_rows: &Rows,
        b_idx: &[usize],
    ) -> Result<(Batch, Rows)> {
        let a_len = a.map_or(0, Top::len);
        let mut indices = Vec::with_capacity(self.keep.min(a_len + b_idx.len()));
        let (mut i, mut j) = (0, 0);
        while indices.len() < self.keep {
            let take_a = match (a.filter(|_| i < a_len), b_idx.get(j)) {
                (Some(a), Some(&bj)) => a.rows.row(i) <= b_rows.row(bj),
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            if take_a {
                indices.push((0, i));
                i += 1;
            } else {
                indices.push((1, b_idx[j]));
                j += 1;
            }
        }
        let mut rows = self.encoder.empty_rows(indices.len(), 0);
        for &(s, r) in &indices {
            match (s, a) {
                (0, Some(a)) => rows.push(a.rows.row(r)),
                _ => rows.push(b_rows.row(r)),
            }
        }
        let sources = match a {
            Some(a) => vec![a.batch.clone(), b.clone()],
            // Source 0 is never referenced without `a`.
            None => vec![b.clone(), b.clone()],
        };
        Ok((interleave_batches(&self.schema, &sources, &indices)?, rows))
    }

    fn retained(&self, rows: usize) {
        self.max_retained.fetch_max(rows as u64, Ordering::Relaxed);
    }
}

struct TopLocal {
    top: Option<Top>,
    reservation: Option<Reservation>,
}

#[derive(Debug, Default)]
struct TopGlobal(Mutex<Option<Top>>);

impl Sink for TopN {
    fn name(&self) -> &'static str {
        "TopN"
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(TopGlobal::default()))
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(TopLocal {
            top: None,
            reservation: Some(ctx.consumer().reservation()),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        if !input.is_unweighted() {
            return Err(KrishivUnsupportedFeature {
                feature: "ORDER BY … LIMIT over weighted (change stream) input".to_owned(),
            }
            .into());
        }
        if self.keep == 0 {
            return Ok(SinkStatus::Finished);
        }
        let l = local_mut::<TopLocal>(local, "TopN")?;
        let rows = self.encoder.encode(input.data())?;
        let threshold = l
            .top
            .as_ref()
            .filter(|t| t.len() >= self.keep)
            .map(|t| t.rows.row(t.len() - 1));
        let mut candidates: Vec<usize> = (0..input.num_rows())
            .filter(|&i| threshold.is_none_or(|t| rows.row(i) < t))
            .collect();
        if candidates.is_empty() {
            return Ok(SinkStatus::NeedMoreInput);
        }
        candidates.sort_unstable_by(|&a, &b| rows.row(a).cmp(&rows.row(b)).then(a.cmp(&b)));
        candidates.truncate(self.keep);
        let (batch, new_rows) = self.merge(l.top.as_ref(), &input, &rows, &candidates)?;
        let bytes = Top::bytes(&batch, &new_rows);
        // Reserve the new state before the old one goes.
        let mut reservation = match l.top.as_mut() {
            Some(t) => t.reservations.pop(),
            None => l.reservation.take(),
        }
        .ok_or_else(|| internal("TopN lost its reservation".to_owned()))?;
        let old = reservation.size();
        if reservation.try_grow(bytes).is_err() {
            match l.top.as_mut() {
                Some(t) => t.reservations.push(reservation),
                None => l.reservation = Some(reservation),
            }
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        reservation.shrink(old);
        self.retained(batch.num_rows());
        l.top = Some(Top {
            batch,
            rows: new_rows,
            reservations: vec![reservation],
        });
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = global_ref::<TopGlobal>(global, "TopN")?;
        let l = local
            .downcast::<TopLocal>()
            .map_err(|_| internal("wrong local state for TopN".to_owned()))?;
        let Some(mine) = l.top else {
            return Ok(());
        };
        let mut shared = lock(&g.0);
        let Some(mut theirs) = shared.take() else {
            *shared = Some(mine);
            return Ok(());
        };
        let all: Vec<usize> = (0..mine.len()).collect();
        let (batch, rows) = self.merge(Some(&theirs), &mine.batch, &mine.rows, &all)?;
        // The merged rows are a subset of both states' rows, so both
        // drivers' reservations cover them until the source is done.
        theirs.reservations.extend(mine.reservations);
        self.retained(batch.num_rows());
        *shared = Some(Top {
            batch,
            rows,
            reservations: theirs.reservations,
        });
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        _global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

/// The source side: the kept rows from `offset`, in batches.
struct TopSource {
    top: Option<Top>,
    pos: usize,
    batch_rows: usize,
    started: bool,
}

impl Source for TopN {
    fn name(&self) -> &'static str {
        "TopN"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        finalized::<TopGlobal>(ctx, "TopN")
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(TopSource {
            top: None,
            pos: self.offset,
            batch_rows: ctx.query().config().batch_size.max(1),
            started: false,
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        let next = (|| {
            let l = local_mut::<TopSource>(local, "TopN")?;
            if !l.started {
                l.started = true;
                l.top = lock(&global_ref::<TopGlobal>(global, "TopN")?.0).take();
            }
            let Some(top) = &l.top else {
                return Ok(None);
            };
            if l.pos >= top.len() {
                // Done: the kept rows and their memory go now.
                l.top = None;
                return Ok(None);
            }
            let n = l.batch_rows.min(top.len() - l.pos);
            let out = top.batch.slice(l.pos, n)?;
            l.pos += n;
            Ok(Some(out))
        })();
        Box::pin(std::future::ready(next))
    }
}
