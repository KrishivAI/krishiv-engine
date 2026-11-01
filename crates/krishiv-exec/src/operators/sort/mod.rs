// SPDX-License-Identifier: Apache-2.0
//! `ORDER BY` (P2-T054): the [`Sort`] breaker with external merge sort and
//! the bounded [`TopN`] for `ORDER BY … LIMIT`.
//!
//! # Ordering (Spark's)
//! Keys are encoded into binary-comparable rows (arrow-row), so a
//! comparison is one `memcmp` whatever the key types. The order is Spark's:
//! - `ASC` defaults to `NULLS FIRST` and `DESC` to `NULLS LAST`
//!   ([`SortKey::asc`], [`SortKey::desc`]); both can be overridden.
//! - Floats: every NaN is equal and greater than every other value, and
//!   `-0.0 = 0.0` (float key columns are canonicalized before encoding; the
//!   output keeps the original values).
//! - Strings and binaries compare by bytes (`UTF8_BINARY`); other
//!   collations are the planner's (it sorts on a collation key column).
//! - Nested key types are refused (`KRISHIV_UNSUPPORTED_FEATURE`).
//!
//! # Stability
//! Within one driver the sort is stable: rows with equal keys come out in
//! the order the driver received them (in-batch ties by row index, runs by
//! the order they were made, merges by run index). Across drivers, equal
//! rows come out grouped in the order the drivers combined, which depends
//! on scheduling (fixed in deterministic mode). Spark makes no promise for
//! ties either: its order depends on partitioning and on spills.
//!
//! # Weights
//! [`Sort`] carries each row's weight (`Weights`) to the output; a
//! constant weight stays constant. [`TopN`] is defined for insert-only
//! input and refuses weighted input.
//!
//! # Work and memory
//! - `sink` encodes and sorts the incoming batch (`O(n log n)` for one
//!   batch, within the ~1 ms bound) and buffers it under a reservation that
//!   covers the batch, its key rows and its order; with a spill scope it
//!   also keeps **headroom** for one spill (an output batch plus the spill
//!   writer's scratch), lent to the writer as grant credit so a spill never
//!   waits for the memory it is about to free.
//! - A spill (requested by the arbitrator, or the driver's own `NeedMemory`)
//!   merges the driver's buffered batches into one sorted run in a spill
//!   file, on the blocking pool ([`TaskContext::run_blocking`]), checking
//!   cancellation between batches.
//! - `finalize`: in memory, nothing to do. Otherwise it spills what is
//!   still buffered, then merges consecutive runs, `fan_in` at a time, on
//!   the blocking pool until at most `fan_in` remain; `fan_in` is the
//!   configured maximum ([`DEFAULT_MAX_MERGE_FAN_IN`]) halved until the
//!   merge's memory (one frame per run plus the writer's scratch) is
//!   granted. The source side streams the final merge of the buffered
//!   batches or the runs, one output batch per blocking-pool call. With a
//!   spill scope an in-memory result keeps its headroom and is a memory
//!   consumer of its own until read ([`crate::output_spill`]): a spill
//!   writes the rest of the merge as one run and the source reads on from
//!   it.
//! - An unbucketed in-memory result of many batches is merged on several
//!   blocking tasks when the memory allows: sampled splitters cut it into one
//!   part per thread, each part is sorted on its own, and the output is
//!   gathered a few batches at a time. The order is the loser tree's, which
//!   runs otherwise. A spill waits for the sorts to finish.
//! - Batches a merge moved past stay referenced until its output batch is
//!   built: in-flight bytes, like every operator's output.

mod keys;
mod merge;
#[expect(
    clippy::module_inception,
    reason = "the Sort operator lives in sort/sort.rs next to TopN"
)]
mod sort;
mod topn;

pub(crate) use keys::KeyEncoder;
pub use keys::SortKey;
pub(crate) use merge::data_bytes;
pub use sort::{DEFAULT_MAX_MERGE_FAN_IN, Sort, SortMetrics};
pub use topn::TopN;

use std::sync::Arc;

use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, internal};
use crate::executor::{Executor, run_blocking};
use crate::operator::GlobalState;

/// A spill writer's write buffer (64 KiB) and file overheads, kept as
/// headroom.
const SPILL_BUFFER_BYTES: usize = 68 << 10;
/// Per-batch allocation slack over its data bytes (buffer alignment and
/// headers), for small spill batches.
const BATCH_SLACK_BYTES: usize = 4 << 10;
/// Target size of a spilled batch: small enough that the headroom to
/// spill (three of them) stays a small part of a memory budget, large
/// enough to amortize a frame's overhead.
const SPILL_BATCH_BYTES: usize = 128 << 10;

/// [`TaskContext::run_blocking`] for a source, which has no context in
/// `get_data`.
pub(crate) async fn blocking<T, F>(
    query: &Arc<QueryContext>,
    executor: &Arc<dyn Executor>,
    f: F,
) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let guard = query.blocking_started();
    run_blocking(&**executor, move || {
        let _guard = guard;
        f()
    })
    .await
}

/// A breaker's source-side global state: its sink's finalized state.
fn finalized<T: Send + Sync + 'static>(ctx: &TaskContext, what: &str) -> Result<GlobalState> {
    let state = ctx
        .dependency()
        .cloned()
        .ok_or_else(|| internal(format!("{what} read before it was built")))?;
    if !state.is::<T>() {
        return Err(internal(format!("wrong global state for {what}")));
    }
    Ok(state)
}

#[cfg(all(test, not(loom)))]
mod tests;
