// SPDX-License-Identifier: Apache-2.0
//! `count_min_sketch(x, eps, confidence, seed)`: Spark's `CountMinSketchAgg` (`CountMinSketchAgg.scala`), over
//! `krishiv_functions::sketch::count_min`.
//!
//! Every group has the same shape, so the counters of all groups are one flat vector (`depth * width` per group,
//! charged as it grows) beside one total per group. NULLs are skipped. The partial state is the serialized sketch,
//! as the result is, so the partial, exchange and final phases need nothing else; a group of no value is the
//! sketch of nothing, never NULL.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, BinaryArray, BinaryViewArray};
use arrow_schema::DataType;
use krishiv_functions::sketch::count_min::Shape;

use super::unsupported;
use crate::error::{Result, internal};

/// The per-group count-min sketches of one aggregate.
#[derive(Debug)]
pub(crate) struct CountMinAcc {
    shape: Shape,
    /// The values each group has counted.
    totals: Vec<i64>,
    /// `shape.cells()` per group.
    counters: Vec<i64>,
}

impl CountMinAcc {
    pub(crate) fn new(depth: u32, width: u32, seed: i32) -> Self {
        Self {
            shape: Shape::new(depth, width, seed),
            totals: Vec::new(),
            counters: Vec::new(),
        }
    }

    /// Adds groups up to `n` with empty sketches.
    pub(crate) fn resize(&mut self, n: usize) {
        self.totals.resize(n, 0);
        self.counters
            .resize(n.saturating_mul(self.shape.cells()), 0);
    }

    /// Room for `new` more groups.
    pub(crate) fn reserve(&mut self, new: usize) {
        self.totals.reserve(new);
        self.counters
            .reserve(new.saturating_mul(self.shape.cells()));
    }

    /// The shape's coefficient buffer, one `i64` per row.
    fn coefficient_bytes(&self) -> usize {
        self.shape.depth().saturating_mul(size_of::<i64>())
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        self.totals
            .capacity()
            .saturating_add(self.counters.capacity())
            .saturating_mul(size_of::<i64>())
            .saturating_add(self.coefficient_bytes())
    }

    /// The size after adding `new` groups, counting the capacity `Vec::reserve` may grow to (twice the current
    /// one, or the length needed).
    pub(crate) fn projected(&self, new: usize) -> usize {
        let grown = |len: usize, cap: usize| len.max(cap.saturating_mul(2));
        let totals = self.totals.len().saturating_add(new);
        let counters = self
            .counters
            .len()
            .saturating_add(new.saturating_mul(self.shape.cells()));
        let after = grown(totals, self.totals.capacity())
            .saturating_add(grown(counters, self.counters.capacity()))
            .saturating_mul(size_of::<i64>())
            .saturating_add(self.coefficient_bytes());
        self.size().max(after)
    }

    /// An upper bound of the serialized partial state of one group.
    pub(crate) fn state_bound(&self) -> usize {
        self.shape
            .cells()
            .saturating_add(self.shape.depth())
            .saturating_mul(size_of::<i64>())
            .saturating_add(24)
    }

    /// The shape beside group `g`'s total and counters.
    fn group(&mut self, g: usize) -> Result<(&Shape, &mut i64, &mut [i64])> {
        let cells = self.shape.cells();
        let total = self.totals.get_mut(g);
        let counters = self
            .counters
            .get_mut(g.saturating_mul(cells)..g.saturating_add(1).saturating_mul(cells));
        match total.zip(counters) {
            Some((total, counters)) => Ok((&self.shape, total, counters)),
            None => Err(internal(
                "a count-min group past the accumulator's groups".to_owned(),
            )),
        }
    }

    /// Counts the rows of `groups` that are `valid` and not NULL, one `add` each.
    fn each<T>(
        &mut self,
        groups: &[u32],
        arg: &dyn Array,
        valid: impl Fn(usize) -> bool,
        value: impl Fn(usize) -> T,
        add: impl Fn(&Shape, &mut [i64], T),
    ) -> Result<()> {
        let nulls = arg.logical_nulls();
        for (i, &g) in groups.iter().enumerate() {
            if !valid(i) || nulls.as_ref().is_some_and(|n| n.is_null(i)) {
                continue;
            }
            if self.shape.has_empty_rows() {
                return Err(unsupported(
                    "count_min_sketch with an infinite or NaN eps (Spark divides by zero)"
                        .to_owned(),
                ));
            }
            let (shape, total, counters) = self.group(g as usize)?;
            add(shape, counters, value(i));
            *total = total.wrapping_add(1);
        }
        Ok(())
    }

    /// Folds one batch of raw values. `valid` is false for a row the `FILTER` mask drops.
    pub(crate) fn update(
        &mut self,
        groups: &[u32],
        arg: &ArrayRef,
        valid: impl Fn(usize) -> bool,
    ) -> Result<()> {
        if arg.len() != groups.len() {
            return Err(internal(
                "a count-min input column of another length than the group ids".to_owned(),
            ));
        }
        macro_rules! integers {
            ($t:ty) => {{
                let a = arg.as_primitive::<$t>();
                self.each(groups, a, valid, |i| i64::from(a.value(i)), Shape::add_long)
            }};
        }
        macro_rules! bytes {
            ($a:expr) => {{
                let a = $a;
                self.each(
                    groups,
                    a,
                    valid,
                    |i| AsRef::<[u8]>::as_ref(a.value(i)),
                    Shape::add_bytes,
                )
            }};
        }
        match arg.data_type() {
            DataType::Null => Ok(()),
            DataType::Int8 => integers!(Int8Type),
            DataType::Int16 => integers!(Int16Type),
            DataType::Int32 => integers!(Int32Type),
            DataType::Int64 => integers!(Int64Type),
            DataType::Utf8 => bytes!(arg.as_string::<i32>()),
            DataType::LargeUtf8 => bytes!(arg.as_string::<i64>()),
            DataType::Utf8View => bytes!(arg.as_string_view()),
            DataType::Binary => bytes!(arg.as_binary::<i32>()),
            DataType::LargeBinary => bytes!(arg.as_binary::<i64>()),
            DataType::BinaryView => bytes!(arg.as_binary_view()),
            other => Err(internal(format!("count_min_sketch over {other:?}"))),
        }
    }

    /// Merges the partial-state sketches into `groups`.
    pub(crate) fn merge(&mut self, groups: &[u32], states: &BinaryArray) -> Result<()> {
        if states.len() != groups.len() {
            return Err(internal(
                "a count-min state of another length than its groups".to_owned(),
            ));
        }
        for (row, &g) in groups.iter().enumerate() {
            if states.is_null(row) {
                continue;
            }
            let (shape, total, counters) = self.group(g as usize)?;
            if !shape.merge(total, counters, states.value(row)) {
                return Err(internal(
                    "a count-min state of another shape than its aggregate".to_owned(),
                ));
            }
        }
        Ok(())
    }

    /// The serialized sketches of groups `ids`.
    fn rows(&self, ids: &[u32]) -> Result<Vec<Vec<u8>>> {
        let cells = self.shape.cells();
        let mut rows = Vec::with_capacity(ids.len());
        let mut bytes = 0_usize;
        for &i in ids {
            let at = i as usize;
            let (Some(&total), Some(counters)) = (
                self.totals.get(at),
                self.counters
                    .get(at.saturating_mul(cells)..at.saturating_add(1).saturating_mul(cells)),
            ) else {
                return Err(internal(
                    "a count-min group past the accumulator's groups".to_owned(),
                ));
            };
            let row = self.shape.serialize(total, counters);
            // A `binary` column's offsets are 32-bit, and `from_iter_values` does not check them.
            bytes = bytes.saturating_add(row.len());
            if i32::try_from(bytes).is_err() {
                return Err(unsupported(
                    "count_min_sketch results of more than 2 GiB in one batch".to_owned(),
                ));
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// The partial state of groups `ids`: the sketches as `binary`.
    pub(crate) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        Ok(vec![
            Arc::new(BinaryArray::from_iter_values(self.rows(ids)?)) as ArrayRef,
        ])
    }

    /// The result of groups `ids`: the sketches on the `binary` view carrier.
    pub(crate) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        Ok(Arc::new(BinaryViewArray::from_iter_values(self.rows(ids)?)) as ArrayRef)
    }
}

#[cfg(test)]
mod tests;
