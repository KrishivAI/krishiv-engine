// SPDX-License-Identifier: Apache-2.0
//! Test-only hooks of the write path, keyed by write id: a closure run at named points of a
//! write (a test cancels the query there), and a weak reference to each write's global state
//! (a test checks it is dropped after the write).

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use krishiv_exec::QueryContext;

/// Where a write calls its hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Point {
    /// A driver wrote a batch.
    Sunk,
    /// `finalize` finished every writer, before the commit.
    Finished,
}

/// A hook: given the point and the write's query; returns whether the write then stalls (a
/// long sleep a cancellation drops).
pub(crate) type Hook = Arc<dyn Fn(Point, &Arc<QueryContext>) -> bool + Send + Sync>;

static HOOKS: Mutex<BTreeMap<String, Hook>> = Mutex::new(BTreeMap::new());

type Globals = BTreeMap<String, Weak<dyn Any + Send + Sync>>;
static GLOBALS: Mutex<Globals> = Mutex::new(BTreeMap::new());

/// Runs `hook` at every point of the write `write_id`.
pub(crate) fn set(write_id: &str, hook: Hook) {
    HOOKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(write_id.to_owned(), hook);
}

/// The write `write_id` reached `point`.
pub(crate) async fn at(write_id: &str, point: Point, query: &Arc<QueryContext>) {
    let hook = HOOKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(write_id)
        .cloned();
    if hook.is_some_and(|h| h(point, query)) {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
    }
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
