// SPDX-License-Identifier: Apache-2.0
//! Parquet Variant binary codec, version 1 (Parquet `VariantEncoding.md`).
//!
//! [`VariantRef`] is a validated borrow over a `(metadata, value)` pair:
//! construction checks every reachable header, count, id and offset, and
//! each child access re-validates its own bounds, so malformed bytes surface
//! as `MALFORMED_VARIANT` instead of panicking. [`VariantBuilder`] turns a
//! JSON document into exactly the bytes Spark 4.2.0's `parse_json` produces
//! (same dictionary order, widths, number rules and error classes).
//!
//! Every later Variant change (the data type, Parquet I/O, `parse_json`,
//! `variant_get`, printing) builds on this codec; nothing else in this
//! module depends on it yet.
//!
//! # Verified behaviour (live Spark 4.2.0 probes, 2026)
//!
//! * Metadata header `0x01 | ((offset_width - 1) << 6)`; object headers carry
//!   the offset width, field-id width and `is_large` bit; arrays likewise
//!   without field ids.
//! * Object field ids are ordered by key bytes with values in input order;
//!   offsets are per-field starts plus the total, and are *not* monotonic.
//! * Integers prefer the smallest int8/16/32/64 that fits; larger integers
//!   become decimals (scale 0) up to 38 digits, then doubles.
//! * Fractions become decimals when `precision <= 38` and `scale <= 38`
//!   (width by `max(precision, scale)`), doubles otherwise; any exponent
//!   means double, with overflow to infinity.
//! * The value buffer is capped at [`VARIANT_SIZE_LIMIT`] bytes with
//!   `VARIANT_SIZE_LIMIT` (size limit rendered as `128.0 MiB`).
//! * Deliberate divergences from Spark are documented on the items involved:
//!   the reader rejects trailing bytes and out-of-range ids that Spark's lax
//!   validator accepts; the builder imposes no nesting-depth cap (Jackson
//!   allows 1000); non-ASCII literal terminators approximate Java's
//!   identifier rule (ASCII, `$` included, is exact).
//!
//! # Unverified
//!
//! Primitive ids 17-19 (time, nanosecond timestamps) follow the encoding
//! text; Spark 4.2.0 reserves them and never emits them, so they are covered
//! only by hand-built byte strings here.

mod builder;
mod reader;

pub use builder::VariantBuilder;
pub use reader::{VariantKind, VariantRef};

/// Maximum Variant value-buffer size in bytes (128 MiB).
///
/// Verified against Spark 4.2.0's `VariantUtil.SIZE_LIMIT`; exceeding it in
/// [`VariantBuilder::from_json`] raises `VARIANT_SIZE_LIMIT`.
pub const VARIANT_SIZE_LIMIT: usize = 134_217_728;

#[cfg(test)]
mod tests;
