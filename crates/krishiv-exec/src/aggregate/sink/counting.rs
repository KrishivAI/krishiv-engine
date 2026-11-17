// SPDX-License-Identifier: Apache-2.0
//! The hash aggregate's hooks for `count(DISTINCT)` counted at the merge
//! ([`crate::aggregate::distinct`]): the key-routed pairs table, the
//! spillable bytes a breaker over two aggregates sums, and the pair counter
//! its final phase counts into.

use std::sync::Arc;
use std::sync::atomic::Ordering as AtomicOrdering;

use super::{GlobalState, HashAggregate, Local, LocalState, global, held, internal, lock};
use crate::aggregate::AggregateSpec;
use crate::aggregate::distinct::PairCounts;
use crate::aggregate::table::AggCore;
use crate::error::Result;

impl HashAggregate {
    /// [`new`](Self::new) whose row layout takes the radix partition bits
    /// of each row's hash from its first `prefix` grouping columns alone
    /// ([`RowLayout::with_route`](crate::rows::RowLayout::with_route)): the
    /// pairs of one key fall in one partition of every driver's table, so
    /// one merge task holds them all (design P3-A 1d). The arrow-row tables
    /// (keys without a row layout) are not routed.
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub(in crate::aggregate) fn routed(spec: AggregateSpec, prefix: usize) -> Result<Arc<Self>> {
        let mut core = AggCore::new(spec)?;
        core.rows = core.rows.map(|l| l.with_route(prefix));
        Ok(Self::of_core(core, None))
    }

    /// The prefix columns that route this aggregate's row hashes (0: none).
    #[cfg(test)]
    pub(in crate::aggregate) fn route(&self) -> usize {
        self.core
            .rows
            .as_ref()
            .map_or(0, crate::rows::RowLayout::route)
    }

    /// What a spill of this driver's state would free, as `sink` reports it
    /// to the driver's spill signal (`None`: the aggregate does not spill).
    /// A breaker over several aggregates reports their sum.
    pub(in crate::aggregate) fn spillable_local(&self, l: &LocalState) -> Option<usize> {
        self.core.spec.spill.as_ref()?;
        let l = l.downcast_ref::<Local>()?;
        let combined = global(&l.global).map_or(0, |g| g.combined.load(AtomicOrdering::Relaxed));
        Some(l.table.memory_size() + combined)
    }

    /// What a spill of the combined tables would free, as `combine` reports
    /// it (`None`: the aggregate does not spill).
    pub(in crate::aggregate) fn spillable_global(&self, g: &GlobalState) -> Option<usize> {
        self.core.spec.spill.as_ref()?;
        Some(held(&lock(&global(g).ok()?.inner)))
    }
}

/// Makes the final phase of the aggregate whose global state is `g` count
/// its finished tables into `counter` ([`crate::aggregate::distinct`]).
///
/// # Errors
/// `INTERNAL_ERROR` for another operator's state, or a second counter.
pub(in crate::aggregate) fn set_pair_counter(
    g: &GlobalState,
    counter: Arc<PairCounts>,
) -> Result<()> {
    global(g)?
        .counter
        .set(counter)
        .map_err(|_| internal("a second pair counter".to_owned()))
}
