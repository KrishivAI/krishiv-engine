// SPDX-License-Identifier: Apache-2.0
//! The query's result: a bounded channel from the root pipeline's drivers
//! to the caller's stream. A full channel accepts the batch and returns
//! `SinkStatus::Blocked`, so drivers stop pulling their sources until the
//! caller reads (backpressure); the channel holds at most its capacity plus
//! one batch per driver. Dropping the stream cancels the query.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use futures::Stream;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_types::batch::Batch;

use crate::context::{QueryContext, TaskContext};
use crate::error::Result;
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus};
use crate::sync::{Mutex, lock, register, wake_all};

#[derive(Debug, Default)]
struct State {
    queue: VecDeque<Batch>,
    /// Set by the query's coordinator once every task has ended.
    end: Option<Option<KrishivError>>,
    receiver_gone: bool,
    receiver: Option<Waker>,
    senders: Vec<Waker>,
}

/// Shared by the result sink and the stream.
#[derive(Debug)]
pub(crate) struct Channel {
    capacity: usize,
    state: Mutex<State>,
}

impl Channel {
    pub(crate) fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity: capacity.max(1),
            state: Mutex::new(State::default()),
        })
    }

    /// Accepts `batch`: `Some(full)`, or `None` if the receiver is gone.
    fn send(&self, batch: Batch) -> Option<bool> {
        let (full, waker) = {
            let mut s = lock(&self.state);
            if s.receiver_gone {
                return None;
            }
            s.queue.push_back(batch);
            (s.queue.len() >= self.capacity, s.receiver.take())
        };
        if let Some(w) = waker {
            w.wake();
        }
        Some(full)
    }

    #[cfg(all(test, loom))]
    pub(crate) fn send_for_test(&self, batch: Batch) -> Option<bool> {
        self.send(batch)
    }

    /// Completes once the queue has room or the receiver is gone.
    pub(crate) fn space(self: Arc<Self>) -> impl Future<Output = ()> + Send + 'static {
        std::future::poll_fn(move |cx| {
            let mut s = lock(&self.state);
            if s.receiver_gone || s.queue.len() < self.capacity {
                return Poll::Ready(());
            }
            register(&mut s.senders, cx.waker());
            Poll::Pending
        })
    }

    /// The coordinator's last word: no more batches; `error` if the query
    /// failed.
    pub(crate) fn close(&self, error: Option<KrishivError>) {
        let waker = {
            let mut s = lock(&self.state);
            s.end = Some(error);
            s.receiver.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }

    pub(crate) fn poll_recv(&self, cx: &mut Context<'_>) -> Poll<Option<Result<Batch>>> {
        let (out, senders) = {
            let mut s = lock(&self.state);
            if let Some(b) = s.queue.pop_front() {
                (Poll::Ready(Some(Ok(b))), std::mem::take(&mut s.senders))
            } else if let Some(end) = s.end.as_mut() {
                // An error is reported once, then the stream ends.
                (Poll::Ready(end.take().map(Err)), Vec::new())
            } else {
                s.receiver = Some(cx.waker().clone());
                (Poll::Pending, Vec::new())
            }
        };
        wake_all(senders);
        out
    }

    pub(crate) fn drop_receiver(&self) {
        let senders = {
            let mut s = lock(&self.state);
            s.receiver_gone = true;
            s.queue.clear();
            std::mem::take(&mut s.senders)
        };
        wake_all(senders);
    }
}

/// The root pipeline's sink.
pub(crate) struct ResultSink {
    pub(crate) channel: Arc<Channel>,
}

impl Sink for ResultSink {
    fn name(&self) -> &'static str {
        "Result"
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(()))
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn sink(
        &self,
        input: Batch,
        _local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        Ok(match self.channel.send(input) {
            None => SinkStatus::Finished,
            Some(false) => SinkStatus::NeedMoreInput,
            Some(true) => SinkStatus::Blocked(Box::pin(Arc::clone(&self.channel).space())),
        })
    }

    fn combine(&self, _local: LocalState, _global: &GlobalState) -> Result<()> {
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        _global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

/// The caller's end: batches as the root pipeline produces them, then the
/// query's error, if any. Dropping it cancels the query.
pub(crate) struct ResultStream {
    pub(crate) channel: Arc<Channel>,
    pub(crate) query: Arc<QueryContext>,
}

impl Stream for ResultStream {
    type Item = Result<Batch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Batch>>> {
        self.channel.poll_recv(cx)
    }
}

impl Drop for ResultStream {
    fn drop(&mut self) {
        // Cancel first, so woken drivers see it.
        self.query.cancel();
        self.channel.drop_receiver();
    }
}
