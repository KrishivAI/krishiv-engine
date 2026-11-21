// SPDX-License-Identifier: Apache-2.0
//! Test-only hooks of the write path, keyed by write id: a closure run at named points of a
//! write (a test cancels the query there, stalls the write or asks the driver to spill), counts
//! of the memory paths a write took, and a weak reference to each write's global state (a test
//! checks it is dropped after the write).

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::Duration;

use krishiv_exec::QueryContext;
use krishiv_memory::{SpillSignal, Spillable as _};

/// Where a write calls its hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Point {
    /// A driver wrote a batch; its writer is still locked.
    Sunk,
    /// The batch's write ended: the driver's seat is idle and its writer unlocked.
    Written,
    /// A driver ended (`combine`): its background release holds the writer, before the flush.
    Combined,
    /// `finalize` finished every writer, before the commit.
    Finished,
}

/// Where the hook was called from.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Site {
    pub(crate) point: Point,
    /// The driver, stable for the write (its writer's address).
    pub(crate) driver: usize,
}

/// What the write does after its hook.
#[derive(Debug, Clone, Default)]
pub(crate) struct Act {
    /// Ask the driver to spill (its own signal) before stalling.
    pub(crate) ask_spill: bool,
    /// Stall (an async sleep, which a cancellation drops).
    pub(crate) stall: Option<Duration>,
    /// Then wait until the flag is set, at most [`WAIT_BOUND`].
    pub(crate) wait_for: Option<Arc<AtomicBool>>,
}

/// The longest an [`Act::wait_for`] waits.
pub(crate) const WAIT_BOUND: Duration = Duration::from_secs(30);

impl Act {
    pub(crate) const NONE: Self = Self {
        ask_spill: false,
        stall: None,
        wait_for: None,
    };

    pub(crate) fn stall(d: Duration) -> Self {
        Self {
            stall: Some(d),
            ..Self::NONE
        }
    }

    pub(crate) fn wait_for(flag: &Arc<AtomicBool>) -> Self {
        Self {
            wait_for: Some(Arc::clone(flag)),
            ..Self::NONE
        }
    }
}

/// A hook: given the site and the write's query.
pub(crate) type Hook = Arc<dyn Fn(Site, &Arc<QueryContext>) -> Act + Send + Sync>;

/// A memory path a write took ([`note`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Path {
    /// A spill request found the writer busy and the batch's write flushed when it ended.
    WantedFlush,
    /// An ended driver's background release shipped buffered row groups.
    CombinedRelease,
}

/// How often a path ran and the bytes it freed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Count {
    pub(crate) times: usize,
    pub(crate) freed: usize,
}

static HOOKS: Mutex<BTreeMap<String, Hook>> = Mutex::new(BTreeMap::new());

static FAILS: Mutex<std::collections::BTreeSet<String>> =
    Mutex::new(std::collections::BTreeSet::new());

/// The write `write_id` fails where it asks [`fails`] (a row-level change's stage, after its
/// listing).
pub(crate) fn fail(write_id: &str) {
    FAILS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(write_id.to_owned());
}

/// Whether the write `write_id` is to fail here.
pub(crate) fn fails(write_id: &str) -> bool {
    FAILS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .contains(write_id)
}

type Globals = BTreeMap<String, Weak<dyn Any + Send + Sync>>;
static GLOBALS: Mutex<Globals> = Mutex::new(BTreeMap::new());

static COUNTS: Mutex<BTreeMap<(String, Path), Count>> = Mutex::new(BTreeMap::new());

/// Runs `hook` at every point of the write `write_id`.
pub(crate) fn set(write_id: &str, hook: Hook) {
    HOOKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(write_id.to_owned(), hook);
}

/// The write `write_id` reached `site`; `signal` is the driver's, when it has one.
pub(crate) async fn at(
    write_id: &str,
    site: Site,
    query: &Arc<QueryContext>,
    signal: Option<&SpillSignal>,
) {
    let hook = HOOKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(write_id)
        .cloned();
    let act = hook.map_or(Act::NONE, |h| h(site, query));
    if act.ask_spill
        && let Some(signal) = signal
    {
        // The driver services it while this future is pending; the ticket is not awaited.
        drop(signal.request_spill(1));
    }
    if let Some(stall) = act.stall {
        tokio::time::sleep(stall).await;
    }
    if let Some(flag) = act.wait_for {
        let deadline = tokio::time::Instant::now() + WAIT_BOUND;
        while !flag.load(Ordering::Acquire) && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
}

/// The write `write_id` took `path`, freeing `freed` bytes.
pub(crate) fn note(write_id: &str, path: Path, freed: usize) {
    let mut counts = COUNTS.lock().unwrap_or_else(PoisonError::into_inner);
    let c = counts.entry((write_id.to_owned(), path)).or_default();
    c.times += 1;
    c.freed += freed;
}

/// How often the write `write_id` took `path`.
pub(crate) fn noted(write_id: &str, path: Path) -> Count {
    COUNTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&(write_id.to_owned(), path))
        .copied()
        .unwrap_or_default()
}

/// Records the global state of the write `write_id`.
pub(crate) fn record<T: Any + Send + Sync>(write_id: &str, global: &Arc<T>) {
    let any: Arc<dyn Any + Send + Sync> = Arc::clone(global) as _;
    GLOBALS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(write_id.to_owned(), Arc::downgrade(&any));
}

/// Whether the global state of the write `write_id` is still alive (`None`: never recorded).
pub(crate) fn global_alive(write_id: &str) -> Option<bool> {
    GLOBALS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(write_id)
        .map(|w| w.strong_count() > 0)
}
