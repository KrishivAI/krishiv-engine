# krishiv-sim

Deterministic simulation testing (DST, ADR 0029): a cluster of simulated workers in one process on
[turmoil](https://github.com/tokio-rs/turmoil) 0.7.2. `publish = false`, layer 7. turmoil was chosen over madsim
in P1-T10 (ADR 0029 update): madsim's upstream is idle and needs forked dependencies.

`simulate(config, setup, client)` builds a turmoil world from a seed: one host per worker, each building its own
deterministic executor, memory pool, exchange and tables inside its host closure, and a coordinator client. Hosts
share only the object store and simulated TCP. Fragments travel as `krishiv.plan.v1` bytes with an unstable task
descriptor (`krishiv.task.v1alpha`), and map outputs as Arrow IPC frames. Map partition `p` runs on worker `p mod W`.
A failed task reruns as its next attempt.

Tests (`tests/tpch_sim.rs`): the 22 TPC-H queries on two workers equal one worker; the same seed gives the same
trace; a map task killed after one write reruns with the same rows. PR CI runs them at SF 0.1
(`KRISHIV_SIM_TPCH_DIR`); locally they read the SF 0.01 fixture.

Not yet: scan splits shipped to workers (P2-T178), runtime filters and other values across stages (P2-T177 unit 3,
row 30c), host crashes and network partitions (P5-T024), SF1 on four workers nightly (P2-T176 unit 3).
