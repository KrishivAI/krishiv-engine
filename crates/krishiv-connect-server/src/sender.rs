// SPDX-License-Identifier: Apache-2.0
//! The response stream of one `ExecutePlan` or `ReattachExecute` (design
//! P3-T011a §3.8; Spark 4.3.0's `ExecuteGrpcResponseSender`, cited as
//! `EGRS`): it attaches to its operation's [`Replay`](crate::operation::Replay)
//! and sends the responses after a given index, in order.
//!
//! - Attaching replaces the sender attached before, which ends with
//!   `INVALID_CURSOR.DISCONNECTED` (`EGRS:284-295`, `ExecuteHolder`
//!   lines 200-213); so does closing the operation.
//! - A reattachable stream ends with OK and no `ResultComplete` once it sent
//!   more than `senderMaxStreamSize` bytes (the response that crosses it is
//!   still sent) or after `senderMaxStreamDuration` (`EGRS:178-191,
//!   217-225, 323-336`); the client then reattaches.
//! - When every response was sent the stream ends, with the operation's
//!   error if it failed (`EGRS:309-322`).
//! - Dropping the stream (the client went away) detaches it. A reattachable
//!   operation keeps running, bounded by its ahead budget, until it is
//!   reattached, released or abandoned; another is closed and removed at
//!   once (`EGRS:127-136`), which cancels its query.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use krishiv_common::error::InvalidCursorDisconnected;
use krishiv_connect::proto::v4_3 as pb;
use tokio::time::Instant;
use tonic::Status;

use crate::execute::ResponseStream;
use crate::operation::replay::Next;
use crate::operation::{ExecuteLimits, Operation, Operations};
use crate::status::status_of;

/// When a reattachable stream ends early for the client to reattach.
#[derive(Debug, Clone, Copy, Default)]
pub struct SenderCaps {
    /// Longest life of one stream; `None`: no limit.
    pub max_duration: Option<Duration>,
    /// Most bytes one stream sends (the crossing response included);
    /// `None`: no limit.
    pub max_size: Option<usize>,
}

impl SenderCaps {
    /// The caps of a stream of `op`: Spark's for a reattachable operation
    /// (zero means none), none for another.
    #[must_use]
    pub fn of(op: &Operation, limits: &ExecuteLimits) -> Self {
        if !op.reattachable() {
            return Self::default();
        }
        Self {
            max_duration: Some(limits.sender_max_stream_duration).filter(|d| !d.is_zero()),
            max_size: Some(limits.sender_max_stream_size).filter(|s| *s > 0),
        }
    }
}

/// One attached sender. Dropping it detaches.
struct Attached {
    op: Arc<Operation>,
    operations: Arc<Operations>,
    generation: u64,
    /// The index of the next response to send.
    next: u64,
    /// Bytes sent so far.
    sent: usize,
    deadline: Option<Instant>,
    max_size: Option<usize>,
    /// The stream sent its last item.
    done: bool,
}

impl Drop for Attached {
    fn drop(&mut self) {
        self.op.replay().detach();
        if !self.op.reattachable() {
            // Spark removes a non-reattachable execution when its one
            // sender ends (`EGRS:132-135`): its query is cancelled if it
            // still runs.
            self.op.close();
            self.operations.remove(&self.op);
        }
    }
}

/// Attaches a new sender to `op` that sends the responses after index
/// `after` (0: from the start), replacing any sender attached before.
pub fn attach(
    op: Arc<Operation>,
    operations: Arc<Operations>,
    after: u64,
    caps: SenderCaps,
) -> ResponseStream {
    let generation = op.replay().attach();
    let now = Instant::now();
    let attached = Attached {
        op,
        operations,
        generation,
        next: after + 1,
        sent: 0,
        deadline: caps.max_duration.and_then(|d| now.checked_add(d)),
        max_size: caps.max_size,
        done: false,
    };
    futures::stream::unfold(attached, step).boxed()
}

async fn step(mut s: Attached) -> Option<(Result<pb::ExecutePlanResponse, Status>, Attached)> {
    if s.done {
        return None;
    }
    let op = Arc::clone(&s.op);
    loop {
        // Registered before the check, so no change in between is missed.
        let changed = op.replay().changed();
        tokio::pin!(changed);
        changed.as_mut().enable();
        let capped = s.max_size.is_some_and(|max| s.sent > max)
            || s.deadline.is_some_and(|d| Instant::now() >= d);
        match op.replay().next(s.generation, s.next, !capped) {
            Next::Response(response, size) => {
                s.next += 1;
                s.sent = s.sent.saturating_add(size);
                return Some((Ok(*response), s));
            }
            Next::Disconnected => {
                s.done = true;
                return Some((Err(status_of(&InvalidCursorDisconnected {}.into())), s));
            }
            Next::Unavailable(e) => {
                s.done = true;
                return Some((Err(status_of(&e)), s));
            }
            Next::Finished(None) => return None,
            Next::Finished(Some(status)) => {
                s.done = true;
                return Some((Err(status), s));
            }
            // A cap is reached and the stream is not finished: end with OK,
            // no `ResultComplete`; the client reattaches.
            Next::Wait if capped => return None,
            Next::Wait => match s.deadline {
                Some(deadline) => {
                    tokio::select! {
                        () = &mut changed => {}
                        () = tokio::time::sleep_until(deadline) => {}
                    }
                }
                None => changed.await,
            },
        }
    }
}
