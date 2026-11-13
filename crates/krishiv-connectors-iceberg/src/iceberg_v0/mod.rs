// SPDX-License-Identifier: Apache-2.0
//! Everything that names an `iceberg` (iceberg-rust 0.11, Arrow 58) item. The module
//! is private: what leaves it is Krishiv types only.

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the commit loop of PR-36a (next commit) calls it")
)]
pub(crate) mod append;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the write path of PR-36a (next commit) calls it")
)]
pub(crate) mod data_file;
pub(crate) mod deletes;
pub(crate) mod eq_deletes;
pub(crate) mod lazy;
pub(crate) mod manifest_cache;
pub(crate) mod provider;
pub(crate) mod scan;
pub(crate) mod stats;
pub(crate) mod storage;
pub(crate) mod table;
pub(crate) mod types;
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the write path of PR-36a (next commit) calls it")
)]
pub(crate) mod write_metrics;

#[cfg(test)]
mod append_tests;
#[cfg(test)]
mod baseline_tests;
#[cfg(test)]
mod data_file_tests;
#[cfg(test)]
mod delete_tests;
#[cfg(test)]
mod dst_tests;
#[cfg(test)]
mod evolution_tests;
#[cfg(test)]
mod provider_tests;
#[cfg(test)]
mod scan_tests;
#[cfg(test)]
mod table_tests;
#[cfg(test)]
mod tests;
