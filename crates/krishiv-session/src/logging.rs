// SPDX-License-Identifier: Apache-2.0
//! Structured logs (ADR 0035, P2-T116): JSON records that carry the query's ids, and a rate limit
//! per callsite. Behind the feature `telemetry-logs`.
//!
//! The engine writes events inside the `krishiv.query` span (`telemetry.rs`); this module is how
//! a binary turns them into records. [`json_layer`] prints each event as one JSON object with the
//! span list, so a record shows `krishiv.query_id`, `krishiv.session_id`, `krishiv.trace_id` and
//! `krishiv.principal` of the query it was written in, and the sanitized `db.query.text` (every
//! literal replaced by `?`), never the statement's text or values. [`EventRateLimit`] keeps one
//! failing query in a loop from filling the disk: each callsite may write a fixed number of events
//! per second, and the rest are counted ([`EventRateLimit::dropped`]), not written. Spans are never
//! limited.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use tracing::Subscriber;
use tracing::metadata::LevelFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::filter::FilterExt as _;
use tracing_subscriber::layer::{Context, Filter, SubscriberExt as _};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt as _;

/// A fixed one-second window per key: the first `limit` calls in a window pass.
#[derive(Debug)]
pub(crate) struct Windows<K> {
    limit: u32,
    seen: Mutex<HashMap<K, (Instant, u32)>>,
}

impl<K: Hash + Eq> Windows<K> {
    pub(crate) fn new(limit: u32) -> Self {
        Self {
            limit,
            seen: Mutex::new(HashMap::new()),
        }
    }

    /// Whether the call for `key` at `now` passes.
    pub(crate) fn admit(&self, key: K, now: Instant) -> bool {
        let mut seen = self.seen.lock().unwrap_or_else(PoisonError::into_inner);
        let slot = seen.entry(key).or_insert((now, 0));
        if now.duration_since(slot.0) >= Duration::from_secs(1) {
            *slot = (now, 0);
        }
        slot.1 = slot.1.saturating_add(1);
        slot.1 <= self.limit
    }
}

/// A per-layer filter that lets each event callsite write `per_second` events per second.
#[derive(Debug)]
pub struct EventRateLimit {
    windows: Windows<tracing::callsite::Identifier>,
    dropped: AtomicU64,
}

impl EventRateLimit {
    /// A limit of `per_second` events per callsite per second.
    #[must_use]
    pub fn new(per_second: u32) -> Self {
        Self {
            windows: Windows::new(per_second),
            dropped: AtomicU64::new(0),
        }
    }

    /// Events dropped so far.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl<S: Subscriber> Filter<S> for EventRateLimit {
    fn enabled(&self, _meta: &tracing::Metadata<'_>, _cx: &Context<'_, S>) -> bool {
        true
    }

    /// Decided per event: the callsite's interest stays "sometimes".
    fn callsite_enabled(
        &self,
        _meta: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }

    fn event_enabled(&self, event: &tracing::Event<'_>, _cx: &Context<'_, S>) -> bool {
        let pass = self
            .windows
            .admit(event.metadata().callsite(), Instant::now());
        if !pass {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        pass
    }
}

/// The JSON layer: one object per event with the fields, the current span and the span list,
/// written to `writer`, at `INFO` and above, at most `per_second` events per callsite per second.
pub fn json_layer<S, W>(per_second: u32, writer: W) -> impl Layer<S>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    W: for<'w> tracing_subscriber::fmt::MakeWriter<'w> + 'static,
{
    tracing_subscriber::fmt::layer()
        .json()
        .with_current_span(true)
        .with_span_list(true)
        .with_writer(writer)
        .with_filter(EventRateLimit::new(per_second).and(LevelFilter::INFO))
}

/// Installs [`json_layer`] on standard error as the process's subscriber.
///
/// # Errors
/// When a global subscriber is already set.
pub fn install_json_logs(per_second: u32) -> Result<(), String> {
    tracing_subscriber::registry()
        .with(json_layer(per_second, std::io::stderr))
        .try_init()
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_passes_its_limit_per_second_and_each_key_has_its_own() {
        let w = Windows::new(3);
        let t0 = Instant::now();
        let a: Vec<bool> = (0..5).map(|_| w.admit("a", t0)).collect();
        assert_eq!(a, [true, true, true, false, false]);
        assert!(w.admit("b", t0), "another key has its own window");
        let later = t0 + Duration::from_millis(1000);
        assert!(w.admit("a", later), "a new window passes again");
        assert!(w.admit("a", later) && w.admit("a", later));
        assert!(!w.admit("a", later));
        assert!(!w.admit("a", later + Duration::from_millis(500)));
    }
}
