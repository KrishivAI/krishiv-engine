// SPDX-License-Identifier: Apache-2.0
//! Scheduler (layer L5): the fragmenter, which cuts a physical plan into
//! stages at its exchanges ([`fragment()`]), and [`StageRunner`], which runs
//! the stages as tasks in this process over a local shuffle (design
//! P2-T176 unit 1). The coordinator across simulated and real hosts, AQE
//! and fault tolerance build on these (P2-T176 unit 2, P5).

#![forbid(unsafe_code)]

mod deal;
mod fragment;
mod stages;

pub use deal::TableDealer;
pub use fragment::{Stage, StageInput, fragment, streaming_scans};
pub use stages::{Partials, StageRunner, StageSplits, merge_partials, produced, stage_tasks};

#[cfg(test)]
mod tests;
