// SPDX-License-Identifier: Apache-2.0
//! The result stream that fills the relation cache as it is read, and what
//! an action keeps of the cache while it runs ([`CacheFill`]).

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_plan::cache::CacheKey;

use crate::cache::{RelationCache, SnapshotLease};
use crate::session::RecordBatchStream;

/// What an action keeps of the relation cache while it runs: the snapshots
/// its plan reads, and what it stores when its stream completes.
pub(crate) struct CacheFill {
    pub(crate) kind: FillKind,
    /// The snapshots the plan reads, held until the action ends.
    pub(crate) lease: SnapshotLease,
}

/// What an action stores into the relation cache when its stream completes.
#[expect(
    clippy::large_enum_variant,
    reason = "one short-lived value per action; boxing adds an allocation for nothing"
)]
pub(crate) enum FillKind {
    /// This frame asked for no cache, or the whole plan already hit.
    No,
    /// Materialize the stream's batches under `key` when it completes.
    Fill {
        key: CacheKey,
        fallback: SchemaRef,
        reads_files: bool,
        tables: Vec<String>,
        /// The cache's invalidation generation before the query was analyzed.
        generation: u64,
        /// The query's own `krishiv.cache.maxBytes`.
        cap: usize,
        /// Whether the registered level may go to disk: the memory cap
        /// does not stop the fill (the snapshot is written to disk).
        disk: bool,
        /// The registration's fill slot: one fill per key runs at a time.
        claim: crate::cache::FillClaim,
    },
}

impl CacheFill {
    /// The snapshots alone, for an action that stores nothing (a write, a
    /// profile): held until the action ends.
    pub(crate) fn into_lease(self) -> SnapshotLease {
        self.lease
    }

    /// `stream`, filling `cache` when it ends cleanly (see [`FillStream`])
    /// and holding the action's snapshots until it is dropped.
    pub(crate) fn fill(
        self,
        stream: RecordBatchStream,
        cache: &Arc<RelationCache>,
    ) -> RecordBatchStream {
        let Self { kind, lease } = self;
        let inner = match kind {
            FillKind::No => stream,
            FillKind::Fill {
                key,
                fallback,
                reads_files,
                tables,
                generation,
                cap,
                disk,
                claim,
            } => Box::pin(FillStream {
                inner: stream,
                cache: Arc::clone(cache),
                key,
                schema: None,
                fallback,
                reads_files,
                tables,
                batches: Vec::new(),
                bytes: 0,
                cap,
                disk,
                generation,
                reservation: Some(cache.reserve_fill()),
                claim: Some(claim),
                filled: false,
                storing: None,
            }),
        };
        Box::pin(Leased {
            inner,
            _lease: lease,
        })
    }
}

/// A stream holding its action's snapshots until it is dropped.
struct Leased {
    inner: RecordBatchStream,
    _lease: SnapshotLease,
}

impl futures::Stream for Leased {
    type Item = Result<RecordBatch, KrishivError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// A result stream that keeps its batches for the relation cache. When the
/// inner stream ends, the batches are stored under the fill's key; a cap
/// overrun of a memory-only level stores nothing, so the query stays
/// uncached instead of failing (a level that uses disk is written to disk
/// instead, and the stream ends once the write has). An error, a
/// cancellation (dropping before the end) or a mid-stream schema change
/// stores nothing either: a cached relation is one complete snapshot, never
/// a prefix of one. The empty result stores an empty snapshot, so later
/// actions read the snapshot, not the source. The buffer is bounded by the
/// cache cap (memory-only levels) and the pool: once the batches kept so far
/// exceed it, they are dropped and the rest of the stream passes through
/// unbuffered.
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
    /// Whether the level may go to disk (the cap does not stop the fill).
    pub(super) disk: bool,
    pub(super) generation: u64,
    /// The pool bytes `batches` hold; dropped (released) when the fill is
    /// abandoned, fails or is cancelled, handed to the cache on success.
    pub(super) reservation: Option<crate::cache::FillReservation>,
    /// The registration's fill slot, held until the fill ends.
    pub(super) claim: Option<crate::cache::FillClaim>,
    pub(super) filled: bool,
    /// The store of the snapshot, once the inner stream ended: the stream
    /// ends when it does (a disk write runs on the blocking pool).
    pub(super) storing: Option<BoxFuture<'static, Option<String>>>,
}

impl FillStream {
    /// Stops buffering: the snapshot cannot be stored.
    fn abandon(&mut self) {
        self.batches = Vec::new();
        self.reservation = None;
        self.claim = None;
        self.schema = None;
        self.filled = true;
    }
}

impl futures::Stream for FillStream {
    type Item = Result<RecordBatch, KrishivError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(storing) = self.storing.as_mut() {
            return match storing.as_mut().poll(cx) {
                Poll::Ready(_) => {
                    self.storing = None;
                    self.claim = None;
                    Poll::Ready(None)
                }
                Poll::Pending => Poll::Pending,
            };
        }
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
                    self.abandon();
                    return Poll::Ready(Some(Ok(batch)));
                }
                let size = crate::cache::batch_bytes(&batch);
                self.bytes = self.bytes.saturating_add(size);
                // Over the cap (a memory-only level; 0 disables caching), or
                // the pool (`krishiv.memory.limit`) refuses the bytes: the
                // snapshot cannot be stored, stop holding it.
                let within = self.cap > 0 && (self.disk || self.bytes <= self.cap);
                let reserved = within && self.reservation.as_mut().is_some_and(|r| r.grow(size));
                if !reserved {
                    self.abandon();
                    return Poll::Ready(Some(Ok(batch)));
                }
                self.batches.push(batch.clone());
                Poll::Ready(Some(Ok(batch)))
            }
            Poll::Ready(None) => {
                if self.filled {
                    return Poll::Ready(None);
                }
                self.filled = true;
                // The empty result still caches (as an empty snapshot),
                // so later actions read the snapshot, not the source.
                let this = &mut *self;
                let Some(reservation) = this.reservation.take() else {
                    return Poll::Ready(None);
                };
                let snapshot = crate::cache::Snapshot {
                    key: this.key.clone(),
                    schema: this
                        .schema
                        .take()
                        .unwrap_or_else(|| Arc::clone(&this.fallback)),
                    batches: std::mem::take(&mut this.batches),
                    reads_files: this.reads_files,
                    tables: std::mem::take(&mut this.tables),
                };
                let limits = crate::cache::FillLimits {
                    generation: this.generation,
                    cap: this.cap,
                    reservation,
                };
                this.storing = Some(Box::pin(Arc::clone(&this.cache).store(snapshot, limits)));
                self.poll_next(cx)
            }
            Poll::Ready(Some(Err(e))) => {
                self.abandon();
                Poll::Ready(Some(Err(e)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
