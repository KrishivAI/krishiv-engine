// SPDX-License-Identifier: Apache-2.0
//! Uploads of encoded files: one PUT for small files, a multipart upload
//! (sized parts, bounded window) for large ones, aborted on error or drop.
//!
//! **Retries.** A single PUT is retried by the scheduler (idempotent). A
//! multipart part is not: `MultipartUpload::put_part` numbers parts by call
//! order, so calling it again would add a part out of order instead of
//! replacing the failed one. Part retries therefore belong to the store
//! client (object_store's cloud clients retry each part request with their
//! `RetryConfig`, which the store registry must keep enabled for uploads);
//! a part that still fails fails the upload, which is then aborted.
//!
//! **Abort ordering.** Parts in flight are drained (completed or failed)
//! before the multipart upload is aborted, so no part lands after the
//! abort and leaves storage behind.

use std::fmt;
use std::sync::Arc;

use super::pace::Direction;
use super::{IoContext, Lane, ScheduledRequest};
use crate::error::{Result, write_failed};
use bytes::Bytes;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use object_store::path::Path;
use object_store::{MultipartUpload, ObjectStore, PutMultipartOptions, PutPayload};

/// Smallest part object stores accept (except the last).
pub const MIN_PART_BYTES: u64 = 5 * 1024 * 1024;
/// Parts in flight per upload.
const MAX_PARTS_IN_FLIGHT: usize = 4;

/// One file being uploaded (see the module docs). Write bytes in file order,
/// then [`finish`](Self::finish) or [`abort`](Self::abort); dropping it
/// unfinished aborts the multipart upload in the background.
pub struct Upload {
    ctx: IoContext,
    store: Arc<dyn ObjectStore>,
    /// The store's pacing and in-flight request cap, shared with reads.
    lane: Arc<Lane>,
    path: Path,
    display: String,
    buf: Vec<Bytes>,
    buf_len: u64,
    written: u64,
    parts: u64,
    multipart: Option<Box<dyn MultipartUpload>>,
    /// Parts being uploaded: tasks on the I/O runtime (or the caller's
    /// runtime), so they progress while the encoder works; polled in place
    /// only outside a runtime. Each yields its byte count.
    in_flight: FuturesUnordered<BoxFuture<'static, Result<u64>>>,
    /// Bytes of the parts in flight (held in memory until they complete).
    in_flight_bytes: u64,
    done: bool,
}

impl fmt::Debug for Upload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Upload")
            .field("path", &self.display)
            .field("written", &self.written)
            .field("parts", &self.parts)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl Upload {
    pub(super) fn new(
        ctx: IoContext,
        store: Arc<dyn ObjectStore>,
        lane: Arc<Lane>,
        path: Path,
        display: String,
    ) -> Self {
        Self {
            ctx,
            store,
            lane,
            path,
            display,
            buf: Vec::new(),
            buf_len: 0,
            written: 0,
            parts: 0,
            multipart: None,
            in_flight: FuturesUnordered::new(),
            in_flight_bytes: 0,
            done: false,
        }
    }

    /// Part size for the next part: the configured size (at least 5 MiB),
    /// doubled every 1,000 parts so a file never needs more than the 10,000
    /// parts object stores allow.
    #[must_use]
    pub fn part_size(&self) -> u64 {
        let base = self.ctx.config().multipart_part_bytes.max(MIN_PART_BYTES);
        base << (self.parts / 1000).min(16)
    }

    /// Bytes accepted and not yet handed to the store.
    #[must_use]
    pub fn buffered(&self) -> u64 {
        self.buf_len
    }

    /// Bytes of parts sent and not yet acknowledged by the store: still held
    /// in memory, so a writer reserves them with [`Self::buffered`].
    #[must_use]
    pub fn in_flight_bytes(&self) -> u64 {
        self.in_flight_bytes
    }

    /// Bytes this upload holds in memory: buffered plus in flight.
    #[must_use]
    pub fn held_bytes(&self) -> u64 {
        self.buf_len + self.in_flight_bytes
    }

    /// Bytes written so far.
    #[must_use]
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Whether the upload window is full (a sink reports `Blocked`).
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.in_flight.len() >= MAX_PARTS_IN_FLIGHT
    }

    /// Appends `bytes`; sends full parts, waiting while the window is full.
    ///
    /// # Errors
    /// `TASK_WRITE_FAILED` if the store refuses a part.
    pub async fn write(&mut self, bytes: Bytes) -> Result<()> {
        self.written += bytes.len() as u64;
        self.buf_len += bytes.len() as u64;
        self.buf.push(bytes);
        while self.buf_len >= self.part_size() {
            let n = self.part_size();
            self.send_part(n).await?;
        }
        Ok(())
    }

    fn take(&mut self, n: u64) -> PutPayload {
        let mut chunks = Vec::new();
        let mut need = n;
        while need > 0 && !self.buf.is_empty() {
            let first = &mut self.buf[0];
            let len = first.len() as u64;
            if len <= need {
                need -= len;
                chunks.push(self.buf.remove(0));
            } else {
                // `need < len`, so it fits in usize.
                let at = usize::try_from(need).unwrap_or(first.len());
                chunks.push(first.split_to(at));
                need = 0;
            }
        }
        self.buf_len -= n - need;
        chunks.into_iter().collect()
    }

    async fn send_part(&mut self, n: u64) -> Result<()> {
        if self.multipart.is_none() {
            let opts = PutMultipartOptions {
                extensions: ScheduledRequest.into_extensions(),
                ..PutMultipartOptions::default()
            };
            // Starting and completing a multipart upload are writes too.
            let lane = Arc::clone(&self.lane);
            let _permit = lane.admit(Direction::Write, &self.path).await?;
            let mp = self
                .store
                .put_multipart_opts(&self.path, opts)
                .await
                .map_err(|e| write_failed(&self.display, &e.to_string()))?;
            self.ctx.count(|c| &c.uploads_started, 1);
            self.multipart = Some(mp);
        }
        let payload = self.take(n);
        let Some(mp) = self.multipart.as_mut() else {
            return Err(write_failed(&self.display, "multipart upload not started"));
        };
        self.ctx.count(|c| &c.requests, 1);
        self.ctx.count(|c| &c.bytes_written, n);
        let part = mp.put_part(payload);
        let shown = self.display.clone();
        let lane = Arc::clone(&self.lane);
        let path = self.path.clone();
        // The part is paced and counts against the store's in-flight cap
        // like a read.
        let part = async move {
            let _permit = lane.admit(Direction::Write, &path).await?;
            part.await
                .map_err(|e| write_failed(&shown, &e.to_string()))
                .map(|()| n)
        };
        let runtime = self
            .ctx
            .scheduler()
            .runtime
            .clone()
            .or_else(|| tokio::runtime::Handle::try_current().ok());
        self.in_flight.push(match runtime {
            Some(rt) => super::spawned(&rt, part),
            None => Box::pin(part),
        });
        self.parts += 1;
        self.in_flight_bytes += n;
        while self.in_flight.len() > MAX_PARTS_IN_FLIGHT {
            self.next_part().await?;
        }
        Ok(())
    }

    async fn next_part(&mut self) -> Result<()> {
        if let Some(r) = self.in_flight.next().await {
            self.in_flight_bytes -= r?;
        }
        Ok(())
    }

    /// Waits until every part sent so far is acknowledged (their memory is
    /// then free).
    ///
    /// # Errors
    /// `TASK_WRITE_FAILED` if a part failed.
    pub async fn wait_idle(&mut self) -> Result<()> {
        while !self.in_flight.is_empty() {
            self.next_part().await?;
        }
        Ok(())
    }

    /// Waits until the window has room.
    ///
    /// # Errors
    /// `TASK_WRITE_FAILED` if a part failed.
    pub async fn wait_capacity(&mut self) -> Result<()> {
        while self.is_full() {
            self.next_part().await?;
        }
        Ok(())
    }

    /// Sends the rest and makes the object visible: one PUT if no part was
    /// sent, else the last part and the multipart completion.
    ///
    /// # Errors
    /// `TASK_WRITE_FAILED`; a started multipart upload is aborted.
    pub async fn finish(mut self) -> Result<u64> {
        let result = self.finish_inner().await;
        if result.is_err() {
            self.abort_inner().await;
        }
        self.done = true;
        result
    }

    async fn finish_inner(&mut self) -> Result<u64> {
        if self.multipart.is_none() {
            let len = self.buf_len;
            let payload = self.take(len);
            self.ctx
                .put(&self.store, &self.lane, &self.path, &self.display, payload)
                .await?;
            return Ok(self.written);
        }
        if self.buf_len > 0 {
            let n = self.buf_len;
            self.send_part(n).await?;
        }
        self.wait_idle().await?;
        let Some(mp) = self.multipart.as_mut() else {
            return Err(write_failed(&self.display, "multipart upload not started"));
        };
        let lane = Arc::clone(&self.lane);
        let _permit = lane.admit(Direction::Write, &self.path).await?;
        mp.complete()
            .await
            .map_err(|e| write_failed(&self.display, &e.to_string()))?;
        self.ctx.count(|c| &c.uploads_completed, 1);
        self.multipart = None;
        Ok(self.written)
    }

    /// Abandons the upload: nothing becomes visible.
    pub async fn abort(mut self) {
        self.abort_inner().await;
        self.done = true;
    }

    async fn abort_inner(&mut self) {
        // Let parts in flight finish (or fail) first: aborting while a part
        // request is still running could let it land after the abort.
        drain(std::mem::take(&mut self.in_flight)).await;
        self.in_flight_bytes = 0;
        self.buf.clear();
        self.buf_len = 0;
        if let Some(mut mp) = self.multipart.take() {
            self.ctx.count(|c| &c.uploads_aborted, 1);
            if let Err(e) = mp.abort().await {
                tracing::warn!(path = self.display, error = %e, "multipart abort failed");
            }
        }
    }
}

impl Drop for Upload {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        if let Some(mut mp) = self.multipart.take() {
            self.ctx.count(|c| &c.uploads_aborted, 1);
            let shown = std::mem::take(&mut self.display);
            let parts = std::mem::take(&mut self.in_flight);
            let abort = async move {
                drain(parts).await;
                if let Err(e) = mp.abort().await {
                    tracing::warn!(path = shown, error = %e, "multipart abort failed");
                }
            };
            match tokio::runtime::Handle::try_current() {
                Ok(h) => {
                    h.spawn(abort);
                }
                Err(_) => tracing::warn!("multipart upload dropped outside a runtime: not aborted"),
            }
        }
    }
}

/// Waits for every part in `parts`, ignoring their outcome.
async fn drain(mut parts: FuturesUnordered<BoxFuture<'static, Result<u64>>>) {
    while parts.next().await.is_some() {}
}
