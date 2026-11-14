// SPDX-License-Identifier: Apache-2.0
//! `Observation` (Spark's `org.apache.spark.sql.Observation`, PySpark's
//! `pyspark.sql.Observation`): the handle `DataFrame::observe` binds to one
//! frame, whose [`Observation::get`] returns the metrics of the first
//! action that ran the observed plan (P3-T084; research
//! `t084-observe-semantics.md` section 3).
//!
//! - One handle observes one frame: a second `observe` with it fails
//!   (`KRISHIV_OBSERVATION_REUSED`; PySpark's `REUSE_OBSERVATION`).
//! - [`Observation::get`] blocks until an action has finished (on success or
//!   on failure, as Spark completes the promise in a `finally`) and never
//!   again changes: the first action wins. Without any action it blocks
//!   forever, as Spark's does; [`Observation::get_timeout`] bounds the wait.
//!   Before `observe` bound the handle it fails
//!   (`KRISHIV_NO_OBSERVE_BEFORE_GET`; PySpark's `NO_OBSERVE_BEFORE_GET`).
//! - The value is one row, a [`RecordBatch`] of one row whose columns are
//!   the metrics, named as written (`count(1)`, `max(v)` when not aliased);
//!   where the observed plan never ran its node (a `LIMIT 0` above it) the
//!   row has no columns (Spark's `{}`).

use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use krishiv_common::error::{
    InvalidObservedMetricsMissingName, KrishivError, KrishivNoObserveBeforeGet,
    KrishivObservationReused,
};

/// Where an observation is.
enum State {
    /// Not bound to a frame by `observe` yet.
    Unbound,
    /// Bound; no action has finished.
    Waiting,
    /// The first action finished.
    Done(RecordBatch),
}

/// What the frames and the handle share.
pub(crate) struct Shared {
    name: String,
    state: Mutex<State>,
    done: Condvar,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The name keying the plan's `CollectMetrics` node.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Binds the handle to a frame.
    pub(crate) fn bind(&self) -> Result<(), KrishivError> {
        let mut state = self.lock();
        match *state {
            State::Unbound => {
                *state = State::Waiting;
                Ok(())
            }
            State::Waiting | State::Done(_) => Err(KrishivObservationReused {
                name: self.name.clone(),
            }
            .into()),
        }
    }

    /// An action finished: `row` is the metrics of the observed node, if it
    /// ran. Only the first action completes the observation.
    pub(crate) fn complete(&self, row: Option<&RecordBatch>) {
        let mut state = self.lock();
        if matches!(*state, State::Waiting) {
            *state = State::Done(
                row.cloned()
                    .unwrap_or_else(|| RecordBatch::new_empty(Arc::new(Schema::empty()))),
            );
            self.done.notify_all();
        }
    }
}

/// The metrics of one observed frame (see the module docs).
///
/// ```
/// # use krishiv_session::{Observation, Session};
/// # use krishiv_session::functions::count;
/// # fn main() -> Result<(), krishiv_common::error::KrishivError> {
/// let s = Session::new()?;
/// let obs = Observation::new();
/// let df = s.range(0, 5, 1, None).observe(&obs, [count(krishiv_session::functions::lit(1)).alias("n")])?;
/// futures::executor::block_on(df.count())?;
/// assert_eq!(obs.get()?.num_rows(), 1);
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct Observation {
    pub(crate) shared: Arc<Shared>,
}

impl std::fmt::Debug for Observation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Observation")
            .field("name", &self.shared.name)
            .finish_non_exhaustive()
    }
}

impl Default for Observation {
    fn default() -> Self {
        Self::new()
    }
}

impl Observation {
    /// An observation named by a fresh UUID (Spark's `Observation()`).
    #[must_use]
    pub fn new() -> Self {
        // Unique within the process, which is all a plan's names need.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self::of(format!("observation-{n}"))
    }

    /// An observation named `name` (Spark's `Observation(name)`): the name
    /// keys the metrics in the plan, and two relations of one plan may not
    /// share it.
    ///
    /// # Errors
    /// `INVALID_OBSERVED_METRICS.MISSING_NAME` for an empty name.
    pub fn named(name: &str) -> Result<Self, KrishivError> {
        if name.is_empty() {
            return Err(InvalidObservedMetricsMissingName {
                operator: "Observation".to_owned(),
            }
            .into());
        }
        Ok(Self::of(name.to_owned()))
    }

    fn of(name: String) -> Self {
        Self {
            shared: Arc::new(Shared {
                name,
                state: Mutex::new(State::Unbound),
                done: Condvar::new(),
            }),
        }
    }

    /// The observation's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.shared.name
    }

    /// The metrics, once an action on the observed frame has finished:
    /// blocks until then. The first action's metrics stay; later actions do
    /// not change them. A frame whose observed node never ran gives a row
    /// without columns.
    ///
    /// # Errors
    /// `KRISHIV_NO_OBSERVE_BEFORE_GET` when no frame was observed with it.
    pub fn get(&self) -> Result<RecordBatch, KrishivError> {
        self.wait(None)?
            .ok_or_else(|| unreachable_wait(&self.shared.name))
    }

    /// As [`Self::get`], giving up after `timeout`: `Ok(None)` if no action
    /// has finished by then.
    ///
    /// # Errors
    /// As [`Self::get`].
    pub fn get_timeout(&self, timeout: Duration) -> Result<Option<RecordBatch>, KrishivError> {
        self.wait(Some(timeout))
    }

    fn wait(&self, timeout: Option<Duration>) -> Result<Option<RecordBatch>, KrishivError> {
        let mut state = self.shared.lock();
        let deadline = timeout.map(|t| std::time::Instant::now() + t);
        loop {
            match &*state {
                State::Unbound => {
                    return Err(KrishivNoObserveBeforeGet {
                        name: self.shared.name.clone(),
                    }
                    .into());
                }
                State::Done(row) => return Ok(Some(row.clone())),
                State::Waiting => {}
            }
            state = match deadline {
                None => self
                    .shared
                    .done
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner),
                Some(deadline) => {
                    let left = deadline.saturating_duration_since(std::time::Instant::now());
                    if left.is_zero() {
                        return Ok(None);
                    }
                    self.shared
                        .done
                        .wait_timeout(state, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0
                }
            };
        }
    }
}

/// `get` waits without a deadline, so `wait` returns a row or an error.
fn unreachable_wait(name: &str) -> KrishivError {
    KrishivNoObserveBeforeGet {
        name: name.to_owned(),
    }
    .into()
}
