// SPDX-License-Identifier: Apache-2.0
//! The rows of a shared plan (P2-T198, design §3.4): a subplan run once per
//! query, kept in a list, and read in full by each [`PhysNode::SharedRead`]
//! (crate::physical::PhysNode::SharedRead).
//!
//! [`SharedSink`] ends the shared plan's pipeline. Each driver collects its
//! batches in memory (charged to its own consumer, spillable: a spill writes
//! them to a file of the driver's) and `combine` hands the pieces to the
//! global state in the order the drivers finish. When the body is ordered
//! (it ends in a sort), the pipeline runs with one driver, so the list keeps
//! the producer's order.
//!
//! `finalize` publishes the list as an [`Output`], a memory consumer of its
//! own whose spill hook ([`OutputSpill`]) writes **every** batch still in
//! memory to files and frees it: with several full readers no batch can be
//! dropped because one reader passed it (this differs from the "part no
//! reader has claimed" of a sort's output). Each reader ([`SharedSource`])
//! has its own cursor over the segments and, for a segment on disk, its own
//! [`SpillReader`] charged to the reader. A batch a reader holds stays alive
//! through its `Arc` when the segment it came from is spilled.
//!
//! The coordinator keeps the sink's finalized state and drops it when the
//! last reading pipeline ended (or the query failed), which drops the list,
//! its memory and its files.

use std::sync::{Arc, Weak};

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::{
    ConsumerHandle, Reservation, SpillFile, SpillReader, SpillScope, SpillSignal,
};
use krishiv_types::batch::Batch;

use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, internal};
use crate::executor::{Executor, run_blocking};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::operators::{global_ref, local_mut};
use crate::output_spill::{OutputSpill, SpillUnread, refused};
use crate::sync::{Mutex, lock};

/// A run of batches of the list: in memory, or in one spill file (in the
/// same order).
enum Segment {
    Memory {
        batches: Vec<Batch>,
        /// The memory the batches are charged to.
        reservation: Reservation,
    },
    File(Arc<SpillFile>),
}

impl Segment {
    fn len(&self) -> u64 {
        match self {
            Self::Memory { batches, .. } => batches.len() as u64,
            Self::File(f) => f.num_batches(),
        }
    }
}

/// What one driver collected: spilled segments first, then the batches in
/// memory.
struct SinkLocal {
    segments: Vec<Segment>,
    batches: Vec<Batch>,
    reservation: Reservation,
    signal: Arc<SpillSignal>,
}

/// The sink side's global state: the segments of every driver, in the order
/// they were combined, and the published list once finalized.
pub(crate) struct SinkGlobal {
    parts: Mutex<Vec<Segment>>,
    done: std::sync::OnceLock<Arc<Output>>,
}

/// The list of a finalized shared plan.
struct Output {
    state: Mutex<Vec<Segment>>,
    schema: SchemaRef,
    /// The hook and consumer of the list, and where a spill writes (with a
    /// spill scope only).
    spill: Option<(Arc<OutputSpill>, Arc<ConsumerHandle>, SpillScope)>,
    executor: Arc<dyn Executor>,
    query: Arc<QueryContext>,
}

impl Output {
    fn batches(&self) -> u64 {
        lock(&self.state).iter().map(Segment::len).sum()
    }

    /// Where the batch `idx` of segment `seg` is.
    fn at(&self, seg: usize, idx: u64) -> Place {
        match lock(&self.state).get(seg) {
            None => Place::End,
            Some(s) if idx >= s.len() => Place::NextSegment,
            Some(Segment::Memory { batches, .. }) => usize::try_from(idx)
                .ok()
                .and_then(|i| batches.get(i))
                .map_or(Place::NextSegment, |b| Place::Memory(b.clone())),
            Some(Segment::File(f)) => Place::File(Arc::clone(f)),
        }
    }

    fn memory_bytes(&self) -> usize {
        lock(&self.state)
            .iter()
            .map(|s| match s {
                Segment::Memory { reservation, .. } => reservation.size(),
                Segment::File(_) => 0,
            })
            .sum()
    }
}

enum Place {
    Memory(Batch),
    File(Arc<SpillFile>),
    NextSegment,
    End,
}

impl SpillUnread for Output {
    fn spill_unread(self: Arc<Self>) -> BoxFuture<'static, Result<usize>> {
        Box::pin(async move {
            let Some((hook, consumer, scope)) = &self.spill else {
                return Ok(0);
            };
            // The batches in memory, by segment. A reader that holds one
            // keeps it alive; readers keep reading the old segment until
            // the file replaces it below.
            let work: Vec<(usize, Vec<Batch>)> = lock(&self.state)
                .iter()
                .enumerate()
                .filter_map(|(i, s)| match s {
                    Segment::Memory { batches, .. } if !batches.is_empty() => {
                        Some((i, batches.clone()))
                    }
                    _ => None,
                })
                .collect();
            if work.is_empty() {
                hook.set_spillable_bytes(0);
                return Ok(0);
            }
            let mut written = Vec::with_capacity(work.len());
            for (i, batches) in work {
                let scratch = consumer.reservation().for_spill_scratch();
                let writer = match scope.writer(Arc::clone(&self.schema), scratch) {
                    Ok(w) => w,
                    Err(e) => return refused(KrishivError::from(e)),
                };
                let query = Arc::clone(&self.query);
                let guard = self.query.blocking_started();
                let file = run_blocking(&*self.executor, move || {
                    let mut writer = writer;
                    for b in &batches {
                        query.check_cancelled()?;
                        writer.write(b).map_err(KrishivError::from)?;
                    }
                    let file = writer.finish().map_err(KrishivError::from)?;
                    Ok((file, guard))
                })
                .await?
                .0;
                written.push((i, Arc::new(file)));
            }
            let mut freed = 0;
            {
                let mut segments = lock(&self.state);
                for (i, file) in written {
                    if let Some(slot) = segments.get_mut(i) {
                        if let Segment::Memory { reservation, .. } = slot {
                            freed += reservation.size();
                        }
                        // Drops the batches and their reservation.
                        *slot = Segment::File(file);
                    }
                }
            }
            hook.set_spillable_bytes(self.memory_bytes());
            Ok(freed)
        })
    }
}

/// The sink ending a shared plan's pipeline.
pub(crate) struct SharedSink {
    schema: SchemaRef,
    scope: Option<SpillScope>,
}

impl SharedSink {
    /// A sink for rows of `schema`; `scope` is where spilled batches go
    /// (without one the list stays in memory).
    pub(crate) fn new(schema: SchemaRef, scope: Option<SpillScope>) -> Arc<Self> {
        Arc::new(Self { schema, scope })
    }
}

impl Sink for SharedSink {
    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(SinkGlobal {
            parts: Mutex::new(Vec::new()),
            done: std::sync::OnceLock::new(),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        let memory = ctx.consumer();
        Ok(Box::new(SinkLocal {
            segments: Vec::new(),
            batches: Vec::new(),
            reservation: memory.reservation(),
            signal: Arc::clone(memory.signal()),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        let l = local_mut::<SinkLocal>(local, "SharedSink")?;
        let bytes = input.data().get_array_memory_size();
        if l.reservation.try_grow(bytes).is_err() {
            return Ok(SinkStatus::NeedMemory { bytes });
        }
        l.batches.push(input);
        if self.scope.is_some() {
            l.signal.set_spillable_bytes(l.reservation.size());
        }
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = global_ref::<SinkGlobal>(global, "SharedSink")?;
        let l = local
            .downcast::<SinkLocal>()
            .map_err(|_| internal("wrong local state for SharedSink".to_owned()))?;
        let SinkLocal {
            segments,
            batches,
            reservation,
            ..
        } = *l;
        let mut parts = lock(&g.parts);
        parts.extend(segments);
        if !batches.is_empty() {
            parts.push(Segment::Memory {
                batches,
                reservation,
            });
        }
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global_ref::<SinkGlobal>(global, "SharedSink")?;
            let mut segments = std::mem::take(&mut *lock(&g.parts));
            let spill = match &self.scope {
                Some(scope) => {
                    // The list gets a consumer of its own: charged to the
                    // finalizing driver's it could not be spilled once that
                    // driver ended.
                    let (hook, consumer) = OutputSpill::register(ctx, "SharedPlan output");
                    for s in &mut segments {
                        if let Segment::Memory { reservation, .. } = s {
                            reservation
                                .transfer_to(&consumer)
                                .map_err(KrishivError::from)?;
                        }
                    }
                    Some((hook, consumer, scope.clone()))
                }
                None => None,
            };
            let out = Arc::new(Output {
                state: Mutex::new(segments),
                schema: Arc::clone(&self.schema),
                spill,
                executor: Arc::clone(ctx.executor()),
                query: Arc::clone(ctx.query()),
            });
            if let Some((hook, _, _)) = &out.spill {
                let weak: Weak<dyn SpillUnread> = Arc::downgrade(&out) as Weak<Output>;
                hook.attach(weak);
                hook.set_spillable_bytes(out.memory_bytes());
            }
            g.done
                .set(out)
                .map_err(|_| internal("a shared plan finalized twice".to_owned()))
        })
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        _target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let Some(scope) = &self.scope else {
                return Ok(0);
            };
            let l = local_mut::<SinkLocal>(local, "SharedSink")?;
            if l.batches.is_empty() {
                return Ok(0);
            }
            let scratch = ctx.consumer().reservation().for_spill_scratch();
            let writer = match scope.writer(Arc::clone(&self.schema), scratch) {
                Ok(w) => w,
                Err(e) => return refused(KrishivError::from(e)),
            };
            let batches = std::mem::take(&mut l.batches);
            let query = Arc::clone(ctx.query());
            let written = ctx
                .run_blocking(move || {
                    let mut writer = writer;
                    for b in &batches {
                        query.check_cancelled()?;
                        writer.write(b).map_err(KrishivError::from)?;
                    }
                    writer
                        .finish()
                        .map_err(KrishivError::from)
                        .map_err(Into::into)
                })
                .await?;
            let freed = l.reservation.size();
            l.reservation.shrink(freed);
            l.signal.set_spillable_bytes(0);
            l.segments.push(Segment::File(Arc::new(written)));
            Ok(freed)
        })
    }
}

/// One `SharedRead`: reads every batch of the list.
pub(crate) struct SharedSource {
    schema: SchemaRef,
}

impl SharedSource {
    /// A reader of rows of `schema` (the shared plan's, in its own order).
    pub(crate) fn new(schema: SchemaRef) -> Arc<Self> {
        Arc::new(Self { schema })
    }
}

/// One reader's global state: its cursor over the list.
struct ReaderGlobal {
    output: Arc<Output>,
    cursor: tokio::sync::Mutex<Cursor>,
    batches: u64,
}

/// One driver of a reader.
struct ReaderLocal {
    consumer: Arc<ConsumerHandle>,
}

struct Cursor {
    segment: usize,
    index: u64,
    /// The reader of a segment on disk, positioned at `index`.
    file: Option<SpillReader>,
}

impl Source for SharedSource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let state = ctx
            .dependency()
            .cloned()
            .ok_or_else(|| internal("a shared plan was read before it was built".to_owned()))?;
        let sink = global_ref::<SinkGlobal>(&state, "SharedSource")?;
        let output = sink
            .done
            .get()
            .cloned()
            .ok_or_else(|| internal("a shared plan was read before it finalized".to_owned()))?;
        let batches = output.batches();
        Ok(Arc::new(ReaderGlobal {
            output,
            cursor: tokio::sync::Mutex::new(Cursor {
                segment: 0,
                index: 0,
                file: None,
            }),
            batches,
        }))
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        global_ref::<ReaderGlobal>(global, "SharedSource").map_or(1, |g| {
            usize::try_from(g.batches).unwrap_or(usize::MAX).max(1)
        })
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        // A spill file this driver opens is read through the driver's own consumer.
        Ok(Box::new(ReaderLocal {
            consumer: Arc::clone(ctx.consumer().consumer()),
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let g = global_ref::<ReaderGlobal>(global, "SharedSource")?;
            let consumer = Arc::clone(&local_mut::<ReaderLocal>(local, "SharedSource")?.consumer);
            let mut c = g.cursor.lock().await;
            loop {
                g.output.query.check_cancelled()?;
                match g.output.at(c.segment, c.index) {
                    Place::End => return Ok(None),
                    Place::NextSegment => {
                        c.segment += 1;
                        c.index = 0;
                        c.file = None;
                    }
                    Place::Memory(b) => {
                        c.index += 1;
                        return Ok(Some(b));
                    }
                    Place::File(file) => {
                        let want = c.index;
                        let reader = c.file.take();
                        let reservation = consumer.reservation();
                        let (batch, reader) = run_blocking(&*g.output.executor, move || {
                            let mut reader = match reader {
                                Some(r) => r,
                                None => {
                                    let mut r =
                                        file.reader(reservation).map_err(KrishivError::from)?;
                                    // A segment spilled under this reader
                                    // continues where it was.
                                    for _ in 0..want {
                                        r.next().transpose().map_err(KrishivError::from)?;
                                    }
                                    r
                                }
                            };
                            let batch = reader.next().transpose().map_err(KrishivError::from)?;
                            Ok((batch, reader))
                        })
                        .await?;
                        c.index += 1;
                        c.file = Some(reader);
                        match batch {
                            Some(b) => return Ok(Some(b)),
                            // The file ended early: a short result is
                            // never returned.
                            None => {
                                return Err(internal(
                                    "a shared plan's spill file ended early".to_owned(),
                                ));
                            }
                        }
                    }
                }
            }
        })
    }
}
