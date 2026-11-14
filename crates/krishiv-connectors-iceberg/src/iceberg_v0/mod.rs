// SPDX-License-Identifier: Apache-2.0
//! Everything that names an `iceberg` (iceberg-rust 0.11, Arrow 58) item. The module
//! is private: what leaves it is Krishiv types only.

pub(crate) mod storage;
pub(crate) mod table;
pub(crate) mod types;

#[cfg(test)]
mod tests;
