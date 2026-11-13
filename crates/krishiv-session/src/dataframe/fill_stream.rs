// SPDX-License-Identifier: Apache-2.0
//! The result stream that fills the relation cache as it is read.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivError;
use krishiv_plan::cache::CacheKey;

use crate::cache::RelationCache;
use crate::session::RecordBatchStream;

/// A result stream that keeps its batches for the relation cache. When the
/// inner stream ends, the batches are stored under the fill's key; a cap
/// overrun stores nothing, so the query stays uncached instead of failing.
/// An error, a cancellation (dropping before the end) or a mid-stream schema
/// change stores nothing either: a cached relation is one complete snapshot,
/// never a prefix of one. The empty result stores an empty snapshot, so
/// later actions read the snapshot, not the source. The buffer is bounded by
/// the cache cap: once the batches kept so far exceed it, they are dropped
/// and the rest of the stream passes through unbuffered.
pub(super) struct FillStream {
    pub(super) inner: RecordBatchStream,
    pub(super) cache: Arc<RelationCache>,
    pub(super) key: CacheKey,
    /// The first batch's schema: what the snapshot is stored as. Operators
    /// may widen the analysis types (large string views, ...), so the cached
    /// table carries the executed schema, as the scan of a created
    /// DataFrame carries the registered one. The analysis schema is only
    /// the fallback for an empty result.
    pub(super) schema: Option<SchemaRef>,
    pub(super) fallback: SchemaRef,
    pub(super) reads_files: bool,
    pub(super) tables: Vec<String>,
    pub(super) batches: Vec<RecordBatch>,
    /// Bytes of `batches`.
    pub(super) bytes: usize,
    /// The fill query's cache cap.
    pub(super) cap: usize,
    pub(super) generation: u64,
    /// The pool bytes `batches` hold; dropped (released) when the fill is
    /// abandoned, fails or is cancelled, handed to the cache on success.
    pub(super) reservation: Option<crate::cache::FillReservation>,
    /// The registration's fill slot, held until the fill ends.
    pub(super) claim: Option<crate::cache::FillClaim>,
    pub(super) filled: bool,
}

impl futures::Stream for FillStream {
    type Item = Result<RecordBatch, KrishivError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                if self.filled {
                    // Abandoned: over the cap or a changed shape.
                    return Poll::Ready(Some(Ok(batch)));
                }
                if self.schema.is_none() {
                    self.schema = Some(Arc::clone(batch.schema_ref()));
                } else if self
                    .schema
                    .as_ref()
                    .is_some_and(|s| s.as_ref() != batch.schema().as_ref())
                {
                    // A stream that changes shape mid-way is not one snapshot.
                    self.batches.clear();
                    self.reservation = None;
                    self.claim = None;
                    self.schema = None;
                    self.filled = true;
                    return Poll::Ready(Some(Ok(batch)));
                }
                let size = crate::cache::batch_bytes(&batch);
                self.bytes = self.bytes.saturating_add(size);
                // Over the cap, or the pool (`krishiv.memory.limit`) refuses
                // the bytes: the snapshot cannot be stored, stop holding it.
                let reserved = self.bytes <= self.cap
                    && self.reservation.as_mut().is_some_and(|r| r.grow(size));
                if !reserved {
                    self.batches = Vec::new();
                    self.reservation = None;
                    self.claim = None;
                    self.schema = None;
                    self.filled = true;
                    return Poll::Ready(Some(Ok(batch)));
                }
                self.batches.push(batch.clone());
                Poll::Ready(Some(Ok(batch)))
            }
            Poll::Ready(None) => {
                if !self.filled {
                    self.filled = true;
                    // The empty result still caches (as an empty snapshot),
                    // so later actions read the snapshot, not the source.
                    let this = &mut *self;
                    let batches = std::mem::take(&mut this.batches);
                    let schema = this
                        .schema
                        .take()
                        .unwrap_or_else(|| Arc::clone(&this.fallback));
                    let cache = Arc::clone(&this.cache);
                    let Some(reservation) = this.reservation.take() else {
                        return Poll::Ready(None);
                    };
                    cache.insert(
                        this.key.clone(),
                        schema,
                        batches,
                        this.reads_files,
                        std::mem::take(&mut this.tables),
                        crate::cache::FillLimits {
                            generation: this.generation,
                            cap: this.cap,
                            reservation,
                        },
                    );
                    this.claim = None;
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(e))) => {
                self.batches.clear();
                self.reservation = None;
                self.claim = None;
                self.filled = true;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
