// SPDX-License-Identifier: Apache-2.0
//! A sort's finished in-memory result ([`Held`]): its buffers and, per
//! bucket, the merge between a reader's steps; spillable until read
//! ([`crate::output_spill`]), a spill writing the rest of each bucket's
//! merge as a run of its own.

use super::*;

/// An in-memory result, spillable until read (see
/// [`crate::output_spill`]): the buffers, and per bucket its merge between
/// the reader's steps. A spill writes the rest of each bucket's merge as a
/// run of its own.
pub(super) struct Held {
    pub(super) state: Mutex<HeldState>,
    /// Notified when a reader's step or a spill gives a merge back.
    pub(super) changed: Event,
    /// The hook and the scope a spill writes to (with a spill scope only).
    pub(super) spill: Option<(Arc<OutputSpill>, SpillScope)>,
    pub(super) schema: SchemaRef,
    pub(super) encoder: Arc<KeyEncoder>,
    /// The result's consumer: the buffers are charged to it.
    pub(super) handle: Arc<ConsumerHandle>,
    pub(super) counters: Arc<Counters>,
    pub(super) query: Arc<QueryContext>,
    pub(super) executor: Arc<dyn Executor>,
    pub(super) batch_rows: usize,
}

pub(super) struct HeldState {
    /// The buffered batches the in-memory buckets read, and the buffers
    /// accounting them: dropped once no bucket reads them.
    pub(super) batches: Arc<[Arc<SortedBatch>]>,
    pub(super) buffers: Vec<Buffer>,
    pub(super) buckets: Vec<Bucket>,
    /// The next bucket a reader claims.
    pub(super) next: usize,
    /// A spill waits for the steps in progress or writes: readers wait.
    pub(super) spilling: bool,
    /// The merge was lost (a failed spill or step): every reader fails
    /// with the error, never with a short result.
    pub(super) failed: Option<KrishivError>,
}

/// One bucket of a [`Held`] result.
pub(super) enum Bucket {
    /// Not opened yet.
    Unread,
    /// The merge between the reader's steps.
    Merging(Box<Merger>),
    /// A reader's step or a spill holds the merge.
    Busy,
    /// The rest of the merge, spilled: the reader takes the run.
    Spilled(SpillRun),
    Done,
}

impl HeldState {
    pub(super) fn failed(e: KrishivError) -> Self {
        Self {
            batches: Arc::new([]),
            buffers: Vec::new(),
            buckets: Vec::new(),
            next: 0,
            spilling: false,
            failed: Some(e),
        }
    }

    /// Whether an in-memory bucket still reads the buffers.
    pub(super) fn reads_buffers(&self) -> bool {
        self.buckets
            .iter()
            .any(|b| matches!(b, Bucket::Unread | Bucket::Merging(_) | Bucket::Busy))
    }
}

/// What a reader's next step on its bucket is.
pub(super) enum Take {
    /// Open the merge of the bucket of these batches.
    Open(Arc<[Arc<SortedBatch>]>),
    /// Continue the merge.
    Merge(Box<Merger>),
    /// Read the rest from this run.
    Run(SpillRun),
    /// A spill holds the result: wait for it.
    Wait,
    /// The bucket is read.
    Done,
}

impl Held {
    pub(super) fn set_spillable(&self, bytes: usize) {
        if let Some((hook, _)) = &self.spill {
            hook.set_spillable_bytes(bytes);
        }
    }

    /// The next bucket for a reader, if any is left; the error once the
    /// result was lost.
    pub(super) fn claim(&self) -> Result<Option<usize>> {
        let mut s = lock(&self.state);
        if let Some(e) = &s.failed {
            return Err(e.clone());
        }
        if s.next >= s.buckets.len() {
            return Ok(None);
        }
        s.next += 1;
        Ok(Some(s.next - 1))
    }

    /// A reader's next step on bucket `b` (leaving it `Busy` for a merge).
    pub(super) fn take(&self, b: usize) -> Result<Take> {
        let mut s = lock(&self.state);
        if let Some(e) = &s.failed {
            return Err(e.clone());
        }
        if s.spilling {
            return Ok(Take::Wait);
        }
        let batches = Arc::clone(&s.batches);
        let Some(bucket) = s.buckets.get_mut(b) else {
            return Err(internal(format!("sort bucket {b} out of range")));
        };
        Ok(match mem::replace(bucket, Bucket::Busy) {
            Bucket::Unread => Take::Open(batches),
            Bucket::Merging(m) => Take::Merge(m),
            Bucket::Spilled(run) => {
                *bucket = Bucket::Done;
                Take::Run(run)
            }
            // A spill holds it.
            Bucket::Busy => Take::Wait,
            Bucket::Done => {
                *bucket = Bucket::Done;
                Take::Done
            }
        })
    }

    /// Ends a step on bucket `b`: `next` is its new state. The buffers go
    /// once no bucket reads them.
    pub(super) fn put(&self, b: usize, next: Bucket) {
        let freed = {
            let mut s = lock(&self.state);
            if let Some(bucket) = s.buckets.get_mut(b) {
                *bucket = next;
            }
            if s.reads_buffers() {
                None
            } else {
                let batches = mem::replace(&mut s.batches, Arc::new([]));
                Some((batches, mem::take(&mut s.buffers)))
            }
        };
        if let Some(freed) = freed {
            self.set_spillable(0);
            drop(freed);
        }
        self.changed.notify();
    }

    /// Fails every reader with `e`; the buffers and merges go.
    pub(super) fn fail(&self, e: KrishivError) {
        let old = mem::replace(&mut *lock(&self.state), HeldState::failed(e));
        self.set_spillable(0);
        drop(old);
        self.changed.notify();
    }
}

/// A bucket's merge taken by a spill, to write: the merge, or `None` to
/// open it.
pub(super) type Unwritten = (usize, Option<Box<Merger>>);

impl SpillUnread for Held {
    fn spill_unread(self: Arc<Self>) -> BoxFuture<'static, Result<usize>> {
        Box::pin(async move {
            let Some((_, scope)) = &self.spill else {
                return Ok(0);
            };
            // Readers stop taking steps; wait for the steps in progress.
            let mut mine = false;
            let (work, batches, mut buffers) = loop {
                let changed = self.changed.listen();
                {
                    let mut s = lock(&self.state);
                    if s.failed.is_some() || (s.spilling && !mine) || !s.reads_buffers() {
                        if mine {
                            s.spilling = false;
                            drop(s);
                            self.changed.notify();
                        }
                        return Ok(0);
                    }
                    s.spilling = true;
                    mine = true;
                    if !s.buckets.iter().any(|b| matches!(b, Bucket::Busy)) {
                        let mut work: Vec<Unwritten> = Vec::new();
                        let batches = Arc::clone(&s.batches);
                        for (i, b) in s.buckets.iter_mut().enumerate() {
                            match mem::replace(b, Bucket::Busy) {
                                // No rows: nothing to write or read.
                                Bucket::Unread
                                    if batches.iter().all(|x| x.bucket(i).is_empty()) =>
                                {
                                    *b = Bucket::Done;
                                }
                                Bucket::Unread => work.push((i, None)),
                                Bucket::Merging(m) => work.push((i, Some(m))),
                                other => *b = other,
                            }
                        }
                        break (work, batches, mem::take(&mut s.buffers));
                    }
                }
                changed.await;
            };
            // Until the buffers are freed, their bytes count as about to be
            // released, so another driver's round that cannot spill waits
            // for them instead of failing. Only from here: the wait above
            // is for the readers' steps, which may arbitrate.
            let _writing = self.handle.spilling();
            // The buffers' headroom is lent to the writers' scratch (as
            // grant credit, so no other consumer takes it meanwhile). The
            // first writer first: if its scratch is refused, nothing was
            // merged yet and every merge goes back as it was.
            let mut freed = 0;
            for b in &mut buffers {
                let lent = b.headroom.size();
                freed += lent + b.reservation.size();
                b.headroom.shrink_to_credit(lent);
            }
            let scratch = self.handle.reservation().for_spill_scratch();
            let writer = match scope.writer(Arc::clone(&self.schema), scratch) {
                Ok(w) => w,
                Err(e) => {
                    self.handle.release_credit();
                    self.give_back(work, buffers);
                    return refused(e.into());
                }
            };
            self.set_spillable(0);
            // Dropped before the spill ends (the query was cancelled): the
            // merges are gone with the blocking job, so readers fail rather
            // than wait on `Busy`.
            let unfinished = Unfinished(Some(&self.state), &self.changed, || {
                HeldState::failed(cancelled())
            });
            let row_bytes = buffers.iter().map(|b| b.row_bytes).max().unwrap_or(1);
            let rows = spill_rows(row_bytes, self.batch_rows);
            let query = Arc::clone(&self.query);
            let guard = self.query.blocking_started();
            let (schema, encoder) = (Arc::clone(&self.schema), Arc::clone(&self.encoder));
            let (handle, scope) = (Arc::clone(&self.handle), scope.clone());
            let written = run_blocking(&*self.executor, move || {
                let out = write_buckets(
                    work, &batches, writer, &scope, &schema, &encoder, &handle, rows, &query,
                );
                // The guard travels with the runs (files), so the query
                // does not end before they are dropped.
                Ok((out, guard))
            })
            .await
            .and_then(|(out, _guard)| out);
            unfinished.disarm();
            self.handle.release_credit();
            match written {
                Ok((runs, rest)) => {
                    for (_, run) in &runs {
                        self.counters.runs.fetch_add(1, Ordering::Relaxed);
                        let bytes = run.file.bytes();
                        self.counters.bytes.fetch_add(bytes, Ordering::Relaxed);
                    }
                    let all = rest.is_empty();
                    {
                        let mut s = lock(&self.state);
                        for (b, run) in runs {
                            if let Some(bucket) = s.buckets.get_mut(b) {
                                *bucket = Bucket::Spilled(run);
                            }
                        }
                    }
                    if all {
                        // Every merge is on disk: the buffers go.
                        drop(buffers);
                        self.give_back(Vec::new(), Vec::new());
                        Ok(freed)
                    } else {
                        // The pool refused a later writer's scratch: the
                        // other buckets stay in memory, with the buffers.
                        self.give_back(rest, buffers);
                        Ok(0)
                    }
                }
                Err(e) => {
                    // Rows already merged are lost: fail the query before
                    // the readers can see the state, and the readers too.
                    self.query.fail(e.clone());
                    self.fail(e.clone());
                    Err(e)
                }
            }
        })
    }
}

impl Held {
    /// Ends a spill: the merges in `work` and `buffers` go back, readers
    /// continue.
    pub(super) fn give_back(&self, work: Vec<Unwritten>, buffers: Vec<Buffer>) {
        let bytes: usize = buffers.iter().map(|b| b.reservation.size()).sum();
        {
            let mut s = lock(&self.state);
            for (b, merger) in work {
                if let Some(bucket) = s.buckets.get_mut(b) {
                    *bucket = merger.map_or(Bucket::Unread, Bucket::Merging);
                }
            }
            s.buffers.extend(buffers);
            s.spilling = false;
            if !s.reads_buffers() {
                s.batches = Arc::new([]);
            }
        }
        self.set_spillable(bytes);
        self.changed.notify();
    }
}

/// The runs a spill wrote (by bucket) and the merges it left in memory.
pub(super) type Written = (Vec<(usize, SpillRun)>, Vec<Unwritten>);

/// Writes the rest of each bucket's merge in `work` as a run of its own
/// (blocking I/O): `writer` for the first, a new one for each next. Returns
/// the runs and, when the pool refused a later writer's scratch, the
/// merges not written.
#[expect(clippy::too_many_arguments, reason = "one blocking job's inputs")]
pub(super) fn write_buckets(
    work: Vec<Unwritten>,
    batches: &[Arc<SortedBatch>],
    writer: krishiv_memory::SpillWriter,
    scope: &SpillScope,
    schema: &SchemaRef,
    encoder: &Arc<KeyEncoder>,
    handle: &Arc<ConsumerHandle>,
    rows: usize,
    query: &QueryContext,
) -> Result<Written> {
    let mut runs = Vec::with_capacity(work.len());
    let mut first = Some(writer);
    let mut work = work.into_iter();
    while let Some((b, merger)) = work.next() {
        let writer = match first.take() {
            Some(w) => w,
            None => {
                match scope.writer(Arc::clone(schema), handle.reservation().for_spill_scratch()) {
                    Ok(w) => w,
                    Err(krishiv_memory::SpillError::Memory(_)) => {
                        let mut rest = vec![(b, merger)];
                        rest.extend(work);
                        return Ok((runs, rest));
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        };
        let mut merger = match merger {
            Some(m) => m,
            None => Box::new(Merger::bucket(
                Arc::clone(schema),
                Arc::clone(encoder),
                batches,
                b,
                handle,
            )?),
        };
        runs.push((b, write_rest(&mut merger, writer, rows, query)?));
    }
    Ok((runs, Vec::new()))
}
