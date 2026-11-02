// SPDX-License-Identifier: Apache-2.0
//! The rounds of a recursive CTE ([`PhysNode::UnionLoop`], T180 unit 5b).
//!
//! [`UnionLoopSource`] is a source: it runs the anchor, then the recursive
//! term again and again, each as a query of its own over a child context of
//! the outer query (its memory pool is a child of the outer query's pool and
//! the outer query's cancellation stops it). Every batch a run produces is
//! handed downstream at once and also appended to the round's working table,
//! an [`Output`] that spills to the query's spill scope under pressure; the
//! next round's [`PhysNode::LoopRef`] reads it. Only the previous round and
//! the one being written are alive, so memory is bounded by two rounds, not
//! by the result.
//!
//! The node runs when its pipeline does: a loop in a branch that is never
//! evaluated never runs. The bounds ([`LoopLimits`]) are plan data.
//!
//! A shared plan (an enclosing non-recursive CTE) read inside the recursive
//! term is built again for every round, as it was when each round was a
//! query of its own.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::StreamExt as _;
use futures::future::BoxFuture;
use krishiv_common::error::{RecursionLevelLimitExceeded, RecursionRowLimitExceeded};
use krishiv_types::batch::Batch;

use crate::context::{OperatorId, QueryContext, QueryId, TaskContext};
use crate::error::{Result, cancelled, internal};
use crate::executor::{Executor, run_blocking};
use crate::operator::{GlobalState, LocalState, Source};
use crate::operators::global_ref;
use crate::physical::{Detached, LoopLimits, PhysNode};
use crate::pipeline::ExecNode;
use crate::runner::QueryRunner;
use crate::shared::Output;

/// Rows that make a run of the working table (one segment, one spill file).
const RUN_ROWS: usize = 8192;

/// Bytes of pending batches that end a run of the working table early, so
/// wide rows in small batches never pile up unreserved.
const RUN_BYTES: usize = 4 << 20;

/// The ids of the queries rounds run as, apart from the outer queries'.
static NEXT_ROUND: AtomicU64 = AtomicU64::new(1 << 62);

/// The source of one `UnionLoop`.
pub(crate) struct UnionLoopSource {
    id: u32,
    schema: SchemaRef,
    anchor: Arc<dyn ExecNode>,
    recursive: Arc<PhysNode>,
    limits: LoopLimits,
    detached: Arc<Detached>,
}

impl UnionLoopSource {
    /// A loop of rows of `schema`: `anchor` built, `recursive` built for each
    /// round through `detached`.
    pub(crate) fn new(
        id: u32,
        schema: SchemaRef,
        anchor: Arc<dyn ExecNode>,
        recursive: PhysNode,
        limits: LoopLimits,
        detached: Arc<Detached>,
    ) -> Arc<Self> {
        Arc::new(Self {
            id,
            schema,
            anchor,
            recursive: Arc::new(recursive),
            limits,
            detached,
        })
    }
}

/// A run in progress: the anchor or one round.
struct Run {
    stream: futures::stream::BoxStream<'static, Result<Batch>>,
    child: Arc<QueryContext>,
}

impl Drop for Run {
    /// A run dropped before its end (downstream stopped reading, or the
    /// outer query ended) stops its child query at its next batch boundary.
    fn drop(&mut self) {
        self.child.cancel();
    }
}

/// The working table being written by a run.
struct Round {
    out: Arc<Output>,
    pending: Vec<Batch>,
    pending_rows: usize,
    pending_bytes: usize,
    rows: u64,
}

struct State {
    run: Option<Run>,
    /// The table the next round reads: the previous run's rows.
    prev: Option<Arc<Output>>,
    next: Option<Round>,
    /// Rows produced so far, anchor included.
    rows: u64,
    /// Rounds started after the anchor.
    level: u32,
    started: bool,
    done: bool,
}

struct LoopGlobal {
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
    op: OperatorId,
    state: tokio::sync::Mutex<State>,
}

impl Source for UnionLoopSource {
    fn name(&self) -> &'static str {
        "UnionLoop"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(LoopGlobal {
            query: Arc::clone(ctx.query()),
            executor: Arc::clone(ctx.executor()),
            op: ctx.op(),
            state: tokio::sync::Mutex::new(State {
                run: None,
                prev: None,
                next: None,
                rows: 0,
                level: 0,
                started: false,
                done: false,
            }),
        }))
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn get_data<'a>(
        &'a self,
        _local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let g = global_ref::<LoopGlobal>(global, "UnionLoop")?;
            let mut st = g.state.lock().await;
            loop {
                g.query.check_cancelled()?;
                if st.done {
                    return Ok(None);
                }
                let Some(run) = st.run.as_mut() else {
                    if !self.begin_run(g, &mut st).await? {
                        return Ok(None);
                    }
                    continue;
                };
                let item = tokio::select! {
                    biased;
                    () = g.query.cancelled() => {
                        run.child.cancel();
                        return Err(cancelled());
                    }
                    item = run.stream.next() => item,
                };
                match item {
                    Some(Err(e)) => return Err(e),
                    Some(Ok(batch)) => {
                        if let Some(b) = self.take_batch(&mut st, batch).await? {
                            return Ok(Some(b));
                        }
                    }
                    None => {
                        if !end_run(&mut st).await? {
                            return Ok(None);
                        }
                    }
                }
            }
        })
    }
}

impl UnionLoopSource {
    /// Starts the anchor or the next round. `false`: the loop ended.
    async fn begin_run(&self, g: &LoopGlobal, st: &mut State) -> Result<bool> {
        let node = if st.started {
            let Some(prev) = st.prev.take() else {
                st.done = true;
                return Ok(false);
            };
            st.level = st.level.saturating_add(1);
            if st.level > self.limits.level {
                return Err(RecursionLevelLimitExceeded {
                    level_limit: self.limits.level.to_string(),
                }
                .into());
            }
            let detached = Arc::clone(&self.detached);
            let recursive = Arc::clone(&self.recursive);
            let loops = HashMap::from([(self.id, prev)]);
            let (node, _schema) = run_blocking(&*g.executor, move || {
                detached.instantiate(&recursive, loops)
            })
            .await?;
            node
        } else {
            st.started = true;
            Arc::clone(&self.anchor)
        };
        let child = QueryContext::new(
            QueryId(NEXT_ROUND.fetch_add(1, Ordering::Relaxed)),
            g.query.class(),
            g.query.config().clone(),
            g.query.pool().child("recursive round", None),
        );
        let stream = QueryRunner::new(Arc::clone(&g.executor))
            .execute(node, Arc::clone(&child))
            .await?;
        st.run = Some(Run { stream, child });
        st.next = Some(Round {
            out: Output::open(
                &g.query,
                &g.executor,
                g.op,
                Arc::clone(&self.schema),
                self.detached.spill().cloned(),
                "recursive working table",
            ),
            pending: Vec::new(),
            pending_rows: 0,
            pending_bytes: 0,
            rows: 0,
        });
        Ok(true)
    }

    /// A batch of the run: relabelled to the loop's schema, counted against
    /// the bounds, kept in the working table. `None`: it was empty.
    async fn take_batch(&self, st: &mut State, batch: Batch) -> Result<Option<Batch>> {
        let data = batch.data();
        let n = data.num_rows();
        if n == 0 {
            return Ok(None);
        }
        let fixed = RecordBatch::try_new(Arc::clone(&self.schema), data.columns().to_vec())
            .map_err(|e| internal(format!("a recursive round: {e}")))?;
        let batch = Batch::new(fixed);
        let n = n as u64;
        st.rows += n;
        let wanted = self.limits.pushed_limit;
        if st.rows > self.limits.rows && wanted.is_none_or(|w| st.rows - n >= w) {
            return Err(RecursionRowLimitExceeded {
                row_limit: self.limits.rows.to_string(),
            }
            .into());
        }
        if wanted.is_some_and(|w| st.rows >= w) {
            // The rows past the limit are never read: stop the run, and wait
            // for its query to finish, so nothing it holds outlives the
            // outer query (dropping the stream only requests cancellation).
            st.done = true;
            st.next = None;
            if let Some(run) = st.run.take() {
                let child = Arc::clone(&run.child);
                drop(run);
                child.finished().await;
            }
            return Ok(Some(batch));
        }
        if let Some(round) = st.next.as_mut() {
            round.rows += n;
            round.pending_rows += batch.data().num_rows();
            round.pending_bytes += batch.data().get_array_memory_size();
            round.pending.push(batch.clone());
            if round.pending_rows >= RUN_ROWS || round.pending_bytes >= RUN_BYTES {
                round.flush().await?;
            }
        }
        Ok(Some(batch))
    }
}

impl Round {
    async fn flush(&mut self) -> Result<()> {
        self.pending_rows = 0;
        self.pending_bytes = 0;
        self.out.append(std::mem::take(&mut self.pending)).await
    }
}

/// The run ended: its rows are the next round's input. `false`: it produced
/// none, the loop is over.
async fn end_run(st: &mut State) -> Result<bool> {
    if let Some(run) = st.run.take() {
        run.child.finished().await;
    }
    let Some(mut round) = st.next.take() else {
        st.done = true;
        return Ok(false);
    };
    if round.rows == 0 {
        st.done = true;
        return Ok(false);
    }
    round.flush().await?;
    st.prev = Some(round.out);
    Ok(true)
}
