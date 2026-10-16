//! Deterministic simulation testing (DST) harness skeleton (P0-T24).
//!
//! The simulator choice (`madsim` vs `turmoil`) is an open question for E01/P1; see the
//! crate README. This crate deliberately has no simulator dependency yet.
#![forbid(unsafe_code)]

/// Configuration shared by every simulated run. The seed makes a run reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimConfig {
    /// Seed for all simulated randomness (scheduling, faults, clocks).
    pub seed: u64,
}

impl SimConfig {
    /// A config with the given seed.
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self::with_seed(0)
    }
}

/// A deterministic simulation backend. Implemented by the madsim or turmoil adapter once chosen.
pub trait Simulation {
    /// Name of the backend, for logs and failure reports ("madsim", "turmoil", ...).
    fn backend(&self) -> &'static str;

    /// The configuration this simulation was built with.
    fn config(&self) -> SimConfig;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(SimConfig);

    impl Simulation for Fake {
        fn backend(&self) -> &'static str {
            "fake"
        }
        fn config(&self) -> SimConfig {
            self.0
        }
    }

    #[test]
    fn seed_round_trips() {
        let sim = Fake(SimConfig::with_seed(42));
        assert_eq!(sim.config().seed, 42);
        assert_eq!(sim.backend(), "fake");
        assert_eq!(SimConfig::default().seed, 0);
    }
}
