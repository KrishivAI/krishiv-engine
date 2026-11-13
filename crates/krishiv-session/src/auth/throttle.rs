// SPDX-License-Identifier: Apache-2.0
//! Failure throttling per peer address (design §3.1; threat model T-A4): after
//! `max_failures` failures within `window`, the address is refused for `block_for`.
//!
//! Addresses are keyed as a client controls them: an IPv4 address (an IPv4-mapped IPv6 one
//! included) whole, an IPv6 address by its /64, since one host usually holds a whole /64 and
//! could otherwise rotate through it.
//!
//! The table is bounded: past [`MAX_PEERS`] keys, stale entries are dropped first (at most one
//! sweep per [`SWEEP_EVERY`], so a full table costs no scan per request); when every entry is
//! live, a new key is not tracked (throttling is a brute-force brake, not an access control, so
//! it degrades open rather than refusing everyone).

use std::collections::HashMap;
use std::net::{IpAddr, Ipv6Addr};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::config::ThrottleConfig;

/// Most peer keys tracked at once.
pub(crate) const MAX_PEERS: usize = 65_536;
/// Least time between two sweeps of a full table.
pub(crate) const SWEEP_EVERY: Duration = Duration::from_secs(1);

/// The throttle key of `ip`: IPv4 whole, IPv6 by its /64.
pub(crate) fn key(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(v4) => IpAddr::V4(v4),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => IpAddr::V4(v4),
            None => IpAddr::V6(Ipv6Addr::from(u128::from(v6) & !u128::from(u64::MAX))),
        },
    }
}

/// The table and when it was last swept.
#[derive(Debug, Default)]
struct Table {
    peers: HashMap<IpAddr, Peer>,
    last_sweep: Option<Instant>,
}

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
    table: Mutex<Table>,
}

impl FailureThrottle {
    /// A throttle with `config`; `max_failures = 0` never blocks.
    #[must_use]
    pub fn new(config: ThrottleConfig) -> Self {
        Self {
            config,
            table: Mutex::new(Table::default()),
        }
    }

    /// Whether `peer` is blocked at `now`.
    #[must_use]
    pub fn is_blocked(&self, peer: IpAddr, now: Instant) -> bool {
        if self.config.max_failures == 0 {
            return false;
        }
        let table = self.table.lock().unwrap_or_else(PoisonError::into_inner);
        table
            .peers
            .get(&key(peer))
            .and_then(|p| p.blocked_until)
            .is_some_and(|until| now < until)
    }

    /// Records a failure from `peer` at `now`; returns whether this failure started a block.
    pub fn record_failure(&self, peer: IpAddr, now: Instant) -> bool {
        if self.config.max_failures == 0 {
            return false;
        }
        let peer = key(peer);
        let mut table = self.table.lock().unwrap_or_else(PoisonError::into_inner);
        if !table.peers.contains_key(&peer) && table.peers.len() >= MAX_PEERS {
            let due = table
                .last_sweep
                .is_none_or(|t| now.saturating_duration_since(t) >= SWEEP_EVERY);
            if due {
                let config = self.config;
                table.peers.retain(|_, p| !is_stale(p, now, config));
                table.last_sweep = Some(now);
            }
            if table.peers.len() >= MAX_PEERS {
                return false;
            }
        }
        let p = table.peers.entry(peer).or_insert(Peer {
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
            // The durations are bounded at parse time; a clock this close to its end blocks
            // for as long as it can.
            let until = now
                .checked_add(self.config.block_for)
                .or_else(|| now.checked_add(Duration::from_hours(24)))
                .unwrap_or(now);
            p.blocked_until = Some(until);
            return until > now;
        }
        false
    }

    /// Addresses tracked now (tests).
    #[cfg(test)]
    pub(crate) fn tracked(&self) -> usize {
        self.table
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .peers
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
pub(crate) fn step(config: ThrottleConfig) -> Duration {
    config.window.max(config.block_for) + Duration::from_secs(1)
}
