// SPDX-License-Identifier: Apache-2.0
//! The bitmap aggregates `bitmap_construct_agg`, `bitmap_or_agg` and `bitmap_and_agg`: Spark's
//! `BitmapConstructAgg`, `BitmapOrAgg` and `BitmapAndAgg` at tag v4.2.0 (`bitmapExpressions.scala`,
//! `BitmapExpressionUtils`; the same at 4.3).
//!
//! A bitmap is [`NUM_BYTES`] bytes (32,768 bits), bit `p % 8` of byte `p / 8`, low bit first. Every
//! group holds one such buffer, in one flat vector (`NUM_BYTES` per group, charged as it grows). The
//! partial state is that buffer itself, so the partial, exchange and final phases need nothing else:
//! construct and OR start from zeros and merge by OR; AND starts from `0xFF` bytes and merges by AND
//! (zeroing the bytes past a shorter input, as Spark's `bitmapAndMerge`). An empty group is the start
//! value (4,096 zero bytes, or `0xFF` bytes for AND), never NULL; NULL inputs are skipped.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BinaryArray, BinaryViewArray, Int64Array};
use krishiv_common::error::InvalidBitmapPosition;

use crate::error::{Result, internal};

/// The bytes of a bitmap (Spark's `BitmapExpressionUtils.NUM_BYTES`).
pub(crate) const NUM_BYTES: usize = 4096;

/// The bits of a bitmap.
const NUM_BITS: i64 = 4096 * 8;

/// Which bitmap aggregate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BitmapOp {
    /// `bitmap_construct_agg`: input is a bit position.
    Construct,
    /// `bitmap_or_agg`: input is a bitmap.
    Or,
    /// `bitmap_and_agg`: input is a bitmap.
    And,
}

/// The per-group bitmaps of one aggregate.
#[derive(Debug)]
pub(crate) struct BitmapAcc {
    op: BitmapOp,
    /// `NUM_BYTES` per group.
    bufs: Vec<u8>,
}

impl BitmapAcc {
    pub(crate) fn new(op: BitmapOp) -> Self {
        Self {
            op,
            bufs: Vec::new(),
        }
    }

    /// A fresh group's byte: zero, or `0xFF` for AND.
    fn start(&self) -> u8 {
        if self.op == BitmapOp::And { 0xFF } else { 0 }
    }

    /// Adds groups up to `n` with empty states.
    pub(crate) fn resize(&mut self, n: usize) {
        let start = self.start();
        self.bufs.resize(n.saturating_mul(NUM_BYTES), start);
    }

    /// Room for `new` more groups.
    pub(crate) fn reserve(&mut self, new: usize) {
        self.bufs.reserve(new.saturating_mul(NUM_BYTES));
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        self.bufs.capacity()
    }

    /// The size after adding `new` groups.
    pub(crate) fn projected(&self, new: usize) -> usize {
        self.size().max(
            self.bufs
                .len()
                .saturating_add(new.saturating_mul(NUM_BYTES)),
        )
    }

    /// Group `g`'s bitmap.
    fn group(&mut self, g: usize) -> Result<&mut [u8]> {
        self.bufs
            .get_mut(g * NUM_BYTES..(g + 1) * NUM_BYTES)
            .ok_or_else(|| internal("a bitmap group past the accumulator's groups".to_owned()))
    }

    /// Sets bit `position` of group `g` (Spark's `BitmapConstructAgg.update`).
    fn set(&mut self, g: usize, position: i64) -> Result<()> {
        if !(0..NUM_BITS).contains(&position) {
            return Err(InvalidBitmapPosition {
                bit_position: position.to_string(),
                bitmap_num_bits: NUM_BITS.to_string(),
                bitmap_num_bytes: NUM_BYTES.to_string(),
            }
            .into());
        }
        let p = usize::try_from(position)
            .map_err(|_| internal("a bitmap position that does not fit".to_owned()))?;
        let byte = self
            .group(g)?
            .get_mut(p / 8)
            .ok_or_else(|| internal("a bitmap position past the buffer".to_owned()))?;
        *byte |= 1 << (p % 8);
        Ok(())
    }

    /// Merges `other` into group `g`: OR, or AND (zeroing the bytes past a shorter `other`).
    fn merge_bytes(&mut self, g: usize, other: &[u8]) -> Result<()> {
        let and = self.op == BitmapOp::And;
        let buf = self.group(g)?;
        if and {
            for (b, &o) in buf.iter_mut().zip(other) {
                *b &= o;
            }
            if other.len() < buf.len() {
                buf[other.len()..].fill(0);
            }
        } else {
            for (b, &o) in buf.iter_mut().zip(other) {
                *b |= o;
            }
        }
        Ok(())
    }

    /// Folds one batch of raw inputs: bit positions for construct, bitmaps otherwise. `valid` is
    /// false for a NULL row and a row the `FILTER` mask drops.
    pub(crate) fn update(
        &mut self,
        groups: &[u32],
        arg: &ArrayRef,
        valid: impl Fn(usize) -> bool,
    ) -> Result<()> {
        if arg.len() != groups.len() {
            return Err(internal(
                "a bitmap input column of another length than the group ids".to_owned(),
            ));
        }
        if self.op == BitmapOp::Construct {
            let Some(positions) = arg.as_any().downcast_ref::<Int64Array>() else {
                return Err(internal(
                    "a bitmap position column of another type than bigint".to_owned(),
                ));
            };
            for (i, &g) in groups.iter().enumerate() {
                if valid(i) && !positions.is_null(i) {
                    self.set(g as usize, positions.value(i))?;
                }
            }
            return Ok(());
        }
        if let Some(a) = arg.as_any().downcast_ref::<BinaryArray>() {
            for (i, &g) in groups.iter().enumerate() {
                if valid(i) && !a.is_null(i) {
                    self.merge_bytes(g as usize, a.value(i))?;
                }
            }
        } else if let Some(a) = arg.as_any().downcast_ref::<BinaryViewArray>() {
            for (i, &g) in groups.iter().enumerate() {
                if valid(i) && !a.is_null(i) {
                    self.merge_bytes(g as usize, a.value(i))?;
                }
            }
        } else {
            return Err(internal(
                "a bitmap column of another type than binary".to_owned(),
            ));
        }
        Ok(())
    }

    /// Merges the partial-state bitmaps into `groups`.
    pub(crate) fn merge(&mut self, groups: &[u32], states: &BinaryArray) -> Result<()> {
        if states.len() != groups.len() {
            return Err(internal(
                "a bitmap state of another length than its groups".to_owned(),
            ));
        }
        for (row, &g) in groups.iter().enumerate() {
            if !states.is_null(row) {
                self.merge_bytes(g as usize, states.value(row))?;
            }
        }
        Ok(())
    }

    /// The bitmaps of groups `ids`.
    fn rows(&self, ids: &[u32]) -> Result<Vec<&[u8]>> {
        let mut rows = Vec::with_capacity(ids.len());
        for &i in ids {
            let at = i as usize * NUM_BYTES;
            rows.push(self.bufs.get(at..at + NUM_BYTES).ok_or_else(|| {
                internal("a bitmap group past the accumulator's groups".to_owned())
            })?);
        }
        Ok(rows)
    }

    /// The partial state of groups `ids`: the bitmaps as `binary`.
    pub(crate) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        Ok(vec![
            Arc::new(BinaryArray::from_iter_values(self.rows(ids)?)) as ArrayRef,
        ])
    }

    /// The result of groups `ids`: the bitmaps on the `binary` view carrier.
    pub(crate) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        Ok(Arc::new(BinaryViewArray::from_iter_values(self.rows(ids)?)) as ArrayRef)
    }
}

#[cfg(test)]
mod tests;
