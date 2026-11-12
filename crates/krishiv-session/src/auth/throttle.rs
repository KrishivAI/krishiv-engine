// SPDX-License-Identifier: Apache-2.0
//! Failure throttling per peer address (design §3.1; threat model T-A4): after
//! `max_failures` failures within `window`, the address is refused for `block_for`.
//!
//! The table is bounded: past [`MAX_PEERS`] addresses, stale entries are dropped first; when
//! every entry is live, a new address is not tracked (throttling is a brute-force brake, not an
//! access control, so it degrades open rather than refusing everyone).

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use super::config::ThrottleConfig;

/// Most peer addresses tracked at once.
pub(crate) const MAX_PEERS: usize = 65_536;

/// One address's state.
#[derive(Debug, Clone, Copy)]
struct Peer {
    window_start: Instant,
    failures: u32,
    blocked_until: Option<Instant>,
}

/// Failure counts per peer address.
#[derive(Debug)]
pub struct FailureThrottle {
    config: ThrottleConfig,
    peers: Mutex<HashMap<IpAddr, Peer>>,
}

impl FailureThrottle {
    /// A throttle with `config`; `max_failures = 0` never blocks.
    #[must_use]
    pub fn new(config: ThrottleConfig) -> Self {
        Self {
            config,
            peers: Mutex::new(HashMap::new()),
        }
    }

    /// Whether `peer` is blocked at `now`.
    #[must_use]
    pub fn is_blocked(&self, peer: IpAddr, now: Instant) -> bool {
        if self.config.max_failures == 0 {
            return false;
        }
        let peers = self.peers.lock().unwrap_or_else(PoisonError::into_inner);
        peers
            .get(&peer)
            .and_then(|p| p.blocked_until)
            .is_some_and(|until| now < until)
    }

    /// Records a failure from `peer` at `now`; returns whether this failure started a block.
    pub fn record_failure(&self, peer: IpAddr, now: Instant) -> bool {
        if self.config.max_failures == 0 {
            return false;
        }
        let mut peers = self.peers.lock().unwrap_or_else(PoisonError::into_inner);
        if !peers.contains_key(&peer) && peers.len() >= MAX_PEERS {
            let config = self.config;
            peers.retain(|_, p| !is_stale(p, now, config));
            if peers.len() >= MAX_PEERS {
                return false;
            }
        }
        let p = peers.entry(peer).or_insert(Peer {
            window_start: now,
            failures: 0,
            blocked_until: None,
        });
        if p.blocked_until.is_some_and(|until| now >= until) {
            p.blocked_until = None;
            p.failures = 0;
            p.window_start = now;
        }
        if now.saturating_duration_since(p.window_start) >= self.config.window {
            p.window_start = now;
            p.failures = 0;
        }
        p.failures = p.failures.saturating_add(1);
        if p.blocked_until.is_none() && p.failures >= self.config.max_failures {
            p.blocked_until = Some(now + self.config.block_for);
            return true;
        }
        false
    }

    /// Addresses tracked now (tests).
    #[cfg(test)]
    pub(crate) fn tracked(&self) -> usize {
        self.peers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// Whether `p` neither blocks nor counts toward a block at `now`.
fn is_stale(p: &Peer, now: Instant, config: ThrottleConfig) -> bool {
    let blocked = p.blocked_until.is_some_and(|until| now < until);
    let counting = now.saturating_duration_since(p.window_start) < config.window;
    !blocked && !counting
}

/// The window a test needs to step past.
#[cfg(test)]
pub(crate) fn step(config: ThrottleConfig) -> std::time::Duration {
    config.window.max(config.block_for) + std::time::Duration::from_secs(1)
}
