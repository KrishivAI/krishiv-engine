// SPDX-License-Identifier: Apache-2.0
//! Fixed-stride key rows in pages, indexed by a salted linear-probing slot
//! table or by `hashbrown` over the same rows (P2-T157 prototype).
//!
//! **Not wired into any operator.** The hash join and the hash aggregate
//! still use [`KeyTable`](crate::hash::KeyTable); this module exists so the
//! two index designs can be measured over one row format before the join
//! (P2-T158) and the aggregate (P2-T159) switch, each behind its own
//! setting. Nothing here is a stable interface.
//!
//! # Rows
//! A [`RowLayout`] fixes, for a list of key types, the bytes of one row:
//!
//! ```text
//! | hash u64 | key validity | pad | views 16 B each | fixed keys, widest first | pad to 8 |
//! ```
//!
//! - Null values are written as zero bytes and padding is zero, so two equal
//!   keys have equal bytes (the rule Polars' key rows use).
//! - Floats are normalized first (every NaN is one NaN, `-0.0` is `0.0`),
//!   as Spark does for grouping and join keys.
//! - Strings and binaries are 16-byte views: the length, then up to 12
//!   bytes inline (zero-padded), or a 4-byte prefix and the offset of the
//!   bytes in the row's page heap. Equality compares the length and prefix
//!   word, then either the inline word or the heap bytes; the offset word
//!   is never compared.
//! - The hash is computed column by column from the values (not from the
//!   row bytes), so `Utf8`, `LargeUtf8` and `Utf8View` carriers of equal
//!   strings hash equally.
//!
//! Rows live in [`RowPages`]: pages of a fixed number of rows, each page
//! with its own heap, addressed by `u32` row ids (`page << shift | row`).
//! Offsets are page-relative, so pages can move, be frozen or be spilled
//! without fixing anything up.
//!
//! # Indexes
//! [`SaltedTable`] is a power-of-two array of `u64` slots, each a 16-bit
//! salt from the top of the hash plus `row id + 1` (`0` is empty), probed
//! linearly; a probe reads the row only when the salt matches (DuckDB's
//! `ht_entry_t`). [`HashbrownTable`] is `hashbrown::HashTable<u32>` of row
//! ids over the same rows. Both look keys up a batch at a time; the salted
//! table first touches every slot of a chunk and then every candidate row,
//! so the cache misses of a chunk overlap, before it resolves the chunk in
//! order.

mod layout;
mod pages;
mod table;

#[cfg(test)]
mod tests;

pub use layout::{EncodedKeys, RowLayout};
pub use pages::RowPages;
pub use table::{HashbrownTable, NO_ROW, RowIndex, SaltedTable};
