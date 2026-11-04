// SPDX-License-Identifier: Apache-2.0
//! The object-store layer shared by the Krishiv engine and platform (ADR 0044).
//!
//! It will hold URL parsing (`s3://`, `gs://`, `abfss://`, ...), store building from
//! Spark-style `fs.*` settings with typed secrets, per-location credentials that expire and
//! refresh, and the retry and pacing policies. Request scheduling, caches and memory
//! accounting stay in `krishiv-io`. The code moves here in later units; this crate is its
//! skeleton.
#![forbid(unsafe_code)]
