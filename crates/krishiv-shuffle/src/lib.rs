// SPDX-License-Identifier: Apache-2.0
//! Shuffle (layer L5): the exchange frame format v1-unstable
//! ([`frame`]) and [`LocalShuffle`], the exchange of the tasks of one
//! process (design P2-T176 unit 1). Remote tiers (Celeborn client, native
//! service) are P5's.

#![forbid(unsafe_code)]

pub mod frame;
mod local;

pub use local::LocalShuffle;

#[cfg(test)]
mod tests;
