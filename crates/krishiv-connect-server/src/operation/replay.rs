// SPDX-License-Identifier: Apache-2.0
//! The responses of one operation (design P3-T011a §3.8; Spark 4.3.0's
//! `ExecuteResponseObserver`, cited below as `ERO`, and the attach and
//! detach bookkeeping of `ExecuteHolder`, `EH`).
//!
//! The encoder pushes responses, numbered from 1; senders consume them by
//! index. A consumed response stays while the responses before it fit the
//! retry buffer (`ERO:281-300`), so a client that lost its stream can
//! reattach after the last response it saw; `release_until` drops them
//! earlier and `close` drops all. Every response id ever produced stays
//! known (`ERO:62-64`, never pruned), so releasing or reattaching after an
//! id that was dropped already still finds its index.
//!
//! A response holds, until it leaves the buffer, its bytes in the memory
//! pool; until it is consumed, its bytes of the operation's ahead budget.
//! The operation's share of the server's response budget (its admission)
//! is held here too, from before its first response until it is closed.
//!
//! One sender is attached at a time (the newest; older ones are
//! disconnected). Its deadline is kept here, so maintenance detaches it at
//! its deadline even when its stream is never polled again
//! ([`Replay::expire_sender`]).
//!
//! One mutex guards the state and is never held across an `await`; a
//! [`Notify`] wakes senders when it changes. Permits and reservations are
//! dropped after the mutex is released.

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use krishiv_common::error::{
    InvalidCursorPositionNotAvailable, InvalidCursorPositionNotFound, KrishivError,
};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_memory::Reservation;
use tokio::sync::{Notify, OwnedSemaphorePermit};
use tonic::Status;

/// What a buffered response holds of the server's budgets.
#[derive(Debug)]
pub struct Held {
    /// Its bytes of the operation's ahead budget, returned once consumed.
    pub ahead: Option<OwnedSemaphorePermit>,
    /// Its bytes in the memory pool (`connect_responses`).
    pub reservation: Option<Reservation>,
}

#[derive(Debug)]
struct Cached {
    response: pb::ExecutePlanResponse,
    size: usize,
    held: Held,
}

/// What [`Replay::next`] found for a sender.
#[derive(Debug)]
pub enum Next {
    /// The response at the asked index and its encoded size.
    Response(Box<pb::ExecutePlanResponse>, usize),
    /// Nothing yet: wait for [`Replay::changed`].
    Wait,
    /// Every response was sent; the stream ends with this error, if any.
    Finished(Option<Status>),
    /// Another sender attached, or the operation was closed: this sender
    /// ends with `INVALID_CURSOR.DISCONNECTED` (`EGRS:284-295`).
    Disconnected,
    /// The response was dropped already
    /// (`INVALID_CURSOR.POSITION_NOT_AVAILABLE`, `ERO:198-204`).
    Unavailable(KrishivError),
}

#[derive(Debug)]
struct State {
    /// Buffered responses, indexes `first..first + len` without gaps.
    responses: VecDeque<Cached>,
    /// The index of `responses[0]`.
    first: u64,
    /// The id of every response produced, by index - 1 (`responseIndexToId`).
    ids: Vec<u128>,
    /// Index of every response id produced (`responseIdToIndex`).
    index_of: HashMap<u128, u64>,
    /// The index of the last response produced (0: none yet).
    last_produced: u64,
    /// Set when the stream is finished: the last index, and the error.
    finished: Option<(u64, Option<Status>)>,
    /// The highest index a sender consumed.
    highest_consumed: u64,
    /// Bumped by every attach: a sender whose generation is older was
    /// replaced.
    generation: u64,
    /// The attached sender: its generation and deadline. `None` once it
    /// detached, was replaced and detached, or passed its deadline.
    current: Option<(u64, Option<Instant>)>,
    /// Since when no sender is attached (Spark's `lastAttachedRpcTimeNs`).
    detached_since: Option<Instant>,
    /// Closed (released or abandoned): every response dropped, senders
    /// disconnected, nothing buffered from now on.
    closed: bool,
    /// Bytes buffered now, and the most at once.
    bytes: usize,
    peak: usize,
    /// The operation's share of the server's response budget.
    admission: Option<OwnedSemaphorePermit>,
}

/// What leaves the buffer under the lock, dropped after it (permits wake
/// waiters, reservations shrink the pool).
#[derive(Default)]
struct Garbage {
    responses: Vec<Cached>,
    held: Vec<Held>,
    permits: Vec<OwnedSemaphorePermit>,
}

impl Garbage {
    fn is_empty(&self) -> bool {
        self.responses.is_empty() && self.held.is_empty() && self.permits.is_empty()
    }
}

/// The responses of one operation and its attached senders.
#[derive(Debug)]
pub struct Replay {
    state: Mutex<State>,
    changed: Notify,
    retry_buffer: usize,
    /// A consumed response is handed out, not kept (a non-reattachable
    /// operation: nobody can ask for it again).
    keep: bool,
}

impl Replay {
    /// An empty buffer that keeps `retry_buffer` bytes of consumed
    /// responses (Spark: 0 for a non-reattachable operation, `ERO:106-110`);
    /// `keep` false hands consumed responses out instead of copying them.
    #[must_use]
    pub fn new(retry_buffer: usize, keep: bool) -> Self {
        Self {
            state: Mutex::new(State {
                responses: VecDeque::new(),
                first: 1,
                ids: Vec::new(),
                index_of: HashMap::new(),
                last_produced: 0,
                finished: None,
                highest_consumed: 0,
                generation: 0,
                current: None,
                detached_since: None,
                closed: false,
                bytes: 0,
                peak: 0,
                admission: None,
            }),
            changed: Notify::new(),
            retry_buffer,
            keep,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Resolves after the next change (a response, the end, an attach, a
    /// release, a close). Create it, `enable` it, then check [`Self::next`],
    /// so no change between the check and the wait is missed.
    pub fn changed(&self) -> tokio::sync::futures::Notified<'_> {
        self.changed.notified()
    }

    /// Buffers `response` (its `response_id` is `id`, the bits of a UUID)
    /// as the next index (`ERO:112-133`). Returns false, dropping it, when
    /// the operation is closed or finished.
    pub fn push(
        &self,
        response: pb::ExecutePlanResponse,
        id: u128,
        size: usize,
        held: Held,
    ) -> bool {
        let mut s = self.lock();
        if s.closed || s.finished.is_some() {
            return false;
        }
        s.last_produced += 1;
        let index = s.last_produced;
        s.ids.push(id);
        s.index_of.insert(id, index);
        s.bytes += size;
        s.peak = s.peak.max(s.bytes);
        s.responses.push_back(Cached {
            response,
            size,
            held,
        });
        drop(s);
        self.changed.notify_waiters();
        true
    }

    /// Marks the stream finished after the last response pushed, with
    /// `error` if it failed (`ERO:151-172`). Once only.
    pub fn finish(&self, error: Option<Status>) {
        let mut s = self.lock();
        if s.finished.is_none() {
            s.finished = Some((s.last_produced, error));
        }
        drop(s);
        self.changed.notify_waiters();
    }

    /// Holds `admission`, the operation's share of the server's response
    /// budget, until [`Self::close`]. Returns false, dropping it, when the
    /// operation is closed already.
    pub fn admit(&self, admission: OwnedSemaphorePermit) -> bool {
        let mut s = self.lock();
        if s.closed {
            drop(s);
            drop(admission);
            return false;
        }
        s.admission = Some(admission);
        true
    }

    /// Attaches a sender whose stream ends at `deadline`, if any
    /// (`EH:200-213`): every sender attached before is replaced. Returns
    /// its generation.
    pub fn attach(&self, deadline: Option<Instant>) -> u64 {
        let mut s = self.lock();
        s.generation += 1;
        let generation = s.generation;
        s.current = Some((generation, deadline));
        s.detached_since = None;
        drop(s);
        self.changed.notify_waiters();
        generation
    }

    /// Detaches the sender of `generation` (`EH:215-224`) when it is still
    /// the attached one: the operation is detached from now on. A replaced
    /// sender detaches nothing.
    pub fn detach(&self, generation: u64) {
        let mut s = self.lock();
        if s.current.is_some_and(|(g, _)| g == generation) {
            s.current = None;
            if !s.closed {
                s.detached_since = Some(Instant::now());
            }
        }
    }

    /// Detaches the attached sender if its deadline passed by `now`
    /// (Spark's sender ends at its deadline even when the client reads
    /// nothing: `EGRS.sendResponse` waits on `isReady` with it). The
    /// operation counts as detached since that deadline. Its stream, if it
    /// is ever polled again, sends nothing more and ends with OK. Returns
    /// whether it detached one.
    pub fn expire_sender(&self, now: Instant) -> bool {
        let mut s = self.lock();
        match s.current {
            Some((_, Some(deadline))) if deadline <= now && !s.closed => {
                s.current = None;
                s.detached_since = Some(deadline);
                true
            }
            _ => false,
        }
    }

    /// Since when no sender is attached; `None` while one is.
    #[must_use]
    pub fn detached_since(&self) -> Option<Instant> {
        self.lock().detached_since
    }

    /// The index of the response `response_id` (`ERO:223-229`).
    ///
    /// # Errors
    /// `INVALID_CURSOR.POSITION_NOT_FOUND` for an id this operation never
    /// produced.
    pub fn index_of(&self, response_id: &str) -> Result<u64, KrishivError> {
        let not_found = || -> KrishivError {
            InvalidCursorPositionNotFound {
                response_id: response_id.to_owned(),
            }
            .into()
        };
        let id = uuid_bits(response_id).ok_or_else(not_found)?;
        self.lock().index_of.get(&id).copied().ok_or_else(not_found)
    }

    /// What the sender of `generation` gets at `index`. With `consume`
    /// false (its caps reached) only the end and an interrupt are reported
    /// (`EGRS:227-281`): the order of checks is Spark's, an interrupt first,
    /// then a response, then the end.
    pub fn next(&self, generation: u64, index: u64, consume: bool) -> Next {
        let mut garbage = Garbage::default();
        let out = self.next_locked(generation, index, consume, &mut garbage);
        if !garbage.is_empty() {
            drop(garbage);
            self.changed.notify_waiters();
        }
        out
    }

    fn next_locked(
        &self,
        generation: u64,
        index: u64,
        consume: bool,
        garbage: &mut Garbage,
    ) -> Next {
        let mut s = self.lock();
        if s.closed || s.generation != generation {
            return Next::Disconnected;
        }
        if consume {
            match self.consume(&mut s, index, garbage) {
                Ok(Some((response, size))) => return Next::Response(Box::new(response), size),
                Ok(None) => {}
                Err(e) => return Next::Unavailable(e),
            }
        }
        // `EGRS.streamFinished`: every response sent, or the observer is
        // cleaned (finished, nothing buffered: `ERO.isCleaned`).
        match &s.finished {
            Some((last, error)) if index > *last || s.responses.is_empty() => {
                Next::Finished(error.clone())
            }
            _ => Next::Wait,
        }
    }

    /// `ERO.consumeResponse` (lines 179-210).
    fn consume(
        &self,
        s: &mut State,
        index: u64,
        garbage: &mut Garbage,
    ) -> Result<Option<(pb::ExecutePlanResponse, usize)>, KrishivError> {
        if index >= s.first && index < s.first + s.responses.len() as u64 {
            let at = usize::try_from(index - s.first).unwrap_or(usize::MAX);
            if index > s.highest_consumed {
                s.highest_consumed = index;
            }
            let out = if self.keep {
                let Some(cached) = s.responses.get_mut(at) else {
                    return Ok(None);
                };
                // Consumed: its bytes leave the ahead budget.
                garbage.permits.extend(cached.held.ahead.take());
                // A cheap copy: the Arrow payload is shared (`Bytes`).
                (cached.response.clone(), cached.size)
            } else {
                // Not reattachable: nobody asks for it again; everything up
                // to it goes.
                let mut taken = None;
                while s.first <= index {
                    let Some(cached) = s.responses.pop_front() else {
                        break;
                    };
                    s.first += 1;
                    s.bytes -= cached.size;
                    garbage.held.push(cached.held);
                    taken = Some((cached.response, cached.size));
                }
                return Ok(taken);
            };
            Self::evict_before(self.retry_buffer, s, index - 1, garbage);
            return Ok(Some(out));
        }
        if index <= s.highest_consumed {
            let id = usize::try_from(index)
                .ok()
                .and_then(|i| i.checked_sub(1))
                .and_then(|i| s.ids.get(i))
                .map_or_else(|| "<UNKNOWN>".to_owned(), |b| uuid_text(*b));
            return Err(InvalidCursorPositionNotAvailable {
                response_id: id,
                index: index.to_string(),
            }
            .into());
        }
        Ok(None)
    }

    /// `ERO.removeCachedResponses` (lines 281-300): walking back from
    /// `last_sent`, keep responses while their total is under the retry
    /// buffer (the one that crosses it too), drop the rest.
    fn evict_before(retry_buffer: usize, s: &mut State, last_sent: u64, garbage: &mut Garbage) {
        let mut i = last_sent;
        let mut total = 0_usize;
        while i >= 1 && i >= s.first && total < retry_buffer {
            let at = usize::try_from(i - s.first).unwrap_or(usize::MAX);
            let Some(cached) = s.responses.get(at) else {
                break;
            };
            total += cached.size;
            i -= 1;
        }
        Self::remove_until(s, i, garbage);
    }

    /// `ERO.removeResponsesUntilIndex` (lines 306-317): drops every buffered
    /// response up to and including `index`.
    fn remove_until(s: &mut State, index: u64, garbage: &mut Garbage) {
        while s.first <= index {
            // `first + len == last_produced + 1` always holds, so an empty
            // buffer means everything up to `index` is gone already.
            let Some(cached) = s.responses.pop_front() else {
                break;
            };
            s.first += 1;
            s.bytes -= cached.size;
            garbage.responses.push(cached);
        }
    }

    /// `ReleaseExecute(release_until)` (`ERO:232-241`): drops the responses
    /// up to and including `response_id`.
    ///
    /// # Errors
    /// `INVALID_CURSOR.POSITION_NOT_FOUND` for an id never produced.
    pub fn release_until(&self, response_id: &str) -> Result<(), KrishivError> {
        let index = self.index_of(response_id)?;
        let mut garbage = Garbage::default();
        let mut s = self.lock();
        Self::remove_until(&mut s, index, &mut garbage);
        drop(s);
        drop(garbage);
        self.changed.notify_waiters();
        Ok(())
    }

    /// Closes the buffer (`EH.close`, lines 283-299): every response
    /// dropped, every sender disconnected, nothing pushed from now on.
    pub fn close(&self) {
        let dropped = {
            let mut s = self.lock();
            s.closed = true;
            s.current = None;
            s.detached_since = None;
            s.bytes = 0;
            s.first += s.responses.len() as u64;
            (std::mem::take(&mut s.responses), s.admission.take())
        };
        // Budgets and reservations return outside the lock.
        drop(dropped);
        self.changed.notify_waiters();
    }

    /// Whether [`Self::close`] ran.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }

    /// How many responses were produced.
    #[must_use]
    pub fn produced(&self) -> u64 {
        self.lock().last_produced
    }

    /// Bytes buffered now.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.lock().bytes
    }

    /// The most bytes buffered at once.
    #[must_use]
    pub fn peak(&self) -> usize {
        self.lock().peak
    }
}

/// The 128 bits of a UUID in its canonical lower-case text form (the form
/// [`super::new_operation_id`] mints); `None` for any other text.
#[must_use]
pub fn uuid_bits(text: &str) -> Option<u128> {
    let b = text.as_bytes();
    if b.len() != 36 {
        return None;
    }
    let mut bits = 0_u128;
    for (i, c) in b.iter().enumerate() {
        if matches!(i, 8 | 13 | 18 | 23) {
            if *c != b'-' {
                return None;
            }
            continue;
        }
        let v = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => return None,
        };
        bits = (bits << 4) | u128::from(v);
    }
    Some(bits)
}

/// The canonical text of a UUID's bits.
#[must_use]
pub fn uuid_text(bits: u128) -> String {
    let hex = format!("{bits:032x}");
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

#[cfg(test)]
mod tests;
