// SPDX-License-Identifier: Apache-2.0
//! Deterministic simulation testing (DST, ADR 0029): a cluster of simulated
//! workers on turmoil in one process (design P2-T176 unit 2).
//!
//! [`simulate`] builds a turmoil world from a seed: one host per worker,
//! each with its own deterministic executor, memory pool, exchange and
//! worker context ([`worker::serve`]), and a coordinator client that gets a
//! [`Coordinator`] (a `krishiv_exec::Scheduler`) and its own worker context.
//! Hosts share nothing but what [`HostSetup`] shares (the object store) and
//! simulated TCP: fragments travel as `krishiv.plan.v1` bytes with task
//! descriptors ([`task`]), map outputs as Arrow IPC frames ([`wire`]).
#![forbid(unsafe_code)]

pub mod cluster;
pub mod exchange;
pub mod task;
pub mod wire;
pub mod worker;

use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::physical::WorkerContext;
use krishiv_exec::{Executor, TokioExecutor};

pub use cluster::Coordinator;
pub use exchange::{Fault, HostExchange, Workers};
pub use worker::HostSetup;

/// The port every simulated worker serves on.
pub const WORKER_PORT: u16 = 7070;

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

/// A deterministic simulation backend (turmoil, ADR 0029).
pub trait Simulation {
    /// Name of the backend, for logs and failure reports.
    fn backend(&self) -> &'static str;

    /// The configuration this simulation was built with.
    fn config(&self) -> SimConfig;
}

/// A simulated cluster's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterConfig {
    /// The run's seed.
    pub sim: SimConfig,
    /// Worker hosts.
    pub workers: usize,
    /// Reduce partitions per hash exchange.
    pub partitions: u32,
    /// Driver slots per pipeline on each host.
    pub slots: usize,
    /// A map task killed once (design §2.7).
    pub fault: Option<Fault>,
}

impl Simulation for ClusterConfig {
    fn backend(&self) -> &'static str {
        "turmoil"
    }

    fn config(&self) -> SimConfig {
        self.sim
    }
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

type HostResult = Result<(), Box<dyn std::error::Error>>;

/// Runs `client` on the coordinator host of a simulated cluster of
/// `config.workers` workers set up by `setup`, and returns the
/// coordinator's trace (see [`Coordinator::trace`]).
///
/// # Errors
/// The client's error, or the simulation's (a host failing, the simulated
/// time running out).
pub fn simulate<F, Fut>(
    config: ClusterConfig,
    setup: Arc<dyn HostSetup>,
    client: F,
) -> Result<Vec<String>, KrishivError>
where
    F: FnOnce(Coordinator, WorkerContext) -> Fut + 'static,
    Fut: Future<Output = Result<(), KrishivError>> + 'static,
{
    let names: Arc<[String]> = (0..config.workers).map(|i| format!("worker-{i}")).collect();
    let workers = Workers {
        names: Arc::clone(&names),
        port: WORKER_PORT,
    };
    let mut sim = turmoil::Builder::new()
        .rng_seed(config.sim.seed)
        .simulation_duration(Duration::from_secs(24 * 3600))
        .build();
    for (index, name) in names.iter().enumerate() {
        let (workers, setup) = (workers.clone(), Arc::clone(&setup));
        sim.host(name.as_str(), move || {
            let (workers, setup) = (workers.clone(), Arc::clone(&setup));
            async move {
                worker::serve(
                    config.sim.seed,
                    index,
                    workers,
                    config.fault,
                    setup,
                    config.slots,
                )
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
        });
    }
    let trace = Arc::new(Mutex::new(Vec::new()));
    let out = Arc::clone(&trace);
    sim.client("coordinator", async move {
        let run = async {
            let executor: Arc<dyn Executor> = TokioExecutor::deterministic(config.slots)?;
            let coordinator = Coordinator::new(executor, workers, config.partitions);
            let worker = setup.worker().await?;
            let outcome = client(coordinator.clone(), worker).await;
            *out.lock().unwrap_or_else(PoisonError::into_inner) = coordinator.trace();
            outcome
        };
        let result: HostResult = run
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>);
        result
    });
    sim.run()
        .map_err(|e| internal(format!("simulation (seed {}): {e}", config.sim.seed)))?;
    let trace = trace.lock().unwrap_or_else(PoisonError::into_inner).clone();
    Ok(trace)
}

/// Helpers of the unit tests: small turmoil worlds with a fake worker.
#[cfg(test)]
pub(crate) mod testing {
    use turmoil::net::TcpListener;

    use super::*;
    use crate::wire::{Reply, Request};

    pub(crate) const PORT: u16 = 7071;

    /// A world with a fixed seed.
    pub(crate) fn sim() -> turmoil::Sim<'static> {
        turmoil::Builder::new().rng_seed(1).build()
    }

    /// The workers `names`, all on [`PORT`].
    pub(crate) fn workers(names: &[&str]) -> Workers {
        Workers {
            names: names.iter().map(|n| (*n).to_owned()).collect(),
            port: PORT,
        }
    }

    async fn answer_forever(reply: fn() -> Reply) -> HostResult {
        let listener = TcpListener::bind(("0.0.0.0", PORT)).await?;
        loop {
            let (mut stream, _) = listener.accept().await?;
            let _ = Request::read(&mut stream).await;
            reply().write(&mut stream).await?;
        }
    }

    /// A host `name` that answers every request with `reply()`.
    pub(crate) fn answering(sim: &mut turmoil::Sim<'_>, name: &'static str, reply: fn() -> Reply) {
        sim.host(name, move || answer_forever(reply));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_round_trips() {
        let config = ClusterConfig {
            sim: SimConfig::with_seed(42),
            workers: 2,
            partitions: 4,
            slots: 2,
            fault: None,
        };
        assert_eq!(config.config().seed, 42);
        assert_eq!(config.backend(), "turmoil");
        assert_eq!(SimConfig::default().seed, 0);
    }

    /// A host whose setup fails after a simulated millisecond, so every
    /// worker and the coordinator get as far as their setup.
    struct Failing;

    impl HostSetup for Failing {
        fn worker(
            &self,
        ) -> futures::future::BoxFuture<'static, Result<WorkerContext, KrishivError>> {
            Box::pin(async {
                tokio::time::sleep(Duration::from_millis(1)).await;
                Err(internal("no tables here".to_owned()))
            })
        }
    }

    #[test]
    fn a_host_that_cannot_set_up_fails_the_simulation() {
        let config = ClusterConfig {
            sim: SimConfig::with_seed(7),
            workers: 2,
            partitions: 2,
            slots: 1,
            fault: None,
        };
        let e = simulate(config, Arc::new(Failing), |_, _| async { Ok(()) }).unwrap_err();
        assert!(
            e.message().contains("simulation (seed 7)"),
            "{}",
            e.message()
        );
        assert!(e.message().contains("no tables here"), "{}", e.message());
    }
}
