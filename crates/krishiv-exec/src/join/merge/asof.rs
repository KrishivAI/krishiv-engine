// SPDX-License-Identifier: Apache-2.0
//! The as-of merge (Spark 4.3's `SortMergeAsOfJoinExec`): within a key's group, each left row
//! takes the nearest right row by the as-of column, among those the residual keeps.
//!
//! Both inputs are sorted on the keys, then the as-of column ascending with NULLs first, so a
//! key's left rows arrive in as-of order and its group (the right rows, buffered or spilled as
//! for any merge) is in as-of order too. A row whose as-of value is NULL never matches: a left
//! one is unmatched, a right one is skipped.
//!
//! Without a residual a slice of left rows walks the group once (two pointers): backward
//! (`L >= R`, `L > R`) keeps the last right row that passes the slice's next left row and gives it
//! to every left row it passes; forward (`L <= R`, `L < R`) gives each left row the first right row
//! that passes it. Ties follow the sort: backward takes the last of equal right values, forward the
//! first, as Spark's scanner does.
//!
//! With a residual the nearest passing row may be any row the comparison allows, so every such pair
//! is evaluated ([`PAIR_BATCH`] at a time): at worst every pair of a key, as in Spark, which also
//! scans from each left row until a row passes. Values compare by their sort rows (canonical
//! floats; structs and arrays in Spark's order).

use std::sync::Arc;

use arrow_array::Array;
use arrow_row::Rows;
use arrow_schema::{DataType, Schema};
use krishiv_types::batch::Batch;

use super::{Merge, PAIR_BATCH, Phase, gather};
use crate::error::{Result, internal};
use crate::operators::{KeyEncoder, SortKey};

/// Which right rows an as-of merge gives a left row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsOfMergeDirection {
    /// The greatest right value at most the left one.
    Backward,
    /// The least right value at least the left one.
    Forward,
}

/// What an as-of merge matches by: the as-of column of each input, of one type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsOfMergeSpec {
    /// The left (probe) input's as-of column.
    pub left_column: usize,
    /// The right (build) input's as-of column.
    pub right_column: usize,
    /// Which right rows match.
    pub direction: AsOfMergeDirection,
    /// Whether an equal value matches.
    pub allow_exact: bool,
}

/// An as-of merge's encoders.
pub(crate) struct AsOfMerge {
    pub(crate) left_column: usize,
    pub(crate) right_column: usize,
    left: Arc<KeyEncoder>,
    right: Arc<KeyEncoder>,
    backward: bool,
    allow_exact: bool,
}

/// Whether sort rows of `a` and `b` compare as one type: equal but for nested field names and
/// nullability, which the row format does not encode. Spark's `(l.ts, l.seq) >= (r.ts, r.seq)`
/// compares structs whose field names are each side's column names.
fn same_layout(a: &DataType, b: &DataType) -> bool {
    match (a, b) {
        (DataType::Struct(x), DataType::Struct(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(p, q)| same_layout(p.data_type(), q.data_type()))
        }
        (DataType::List(x), DataType::List(y))
        | (DataType::LargeList(x), DataType::LargeList(y)) => {
            same_layout(x.data_type(), y.data_type())
        }
        (DataType::FixedSizeList(x, n), DataType::FixedSizeList(y, m)) => {
            n == m && same_layout(x.data_type(), y.data_type())
        }
        _ => a == b,
    }
}

impl AsOfMerge {
    pub(crate) fn new(left: &Schema, right: &Schema, spec: AsOfMergeSpec) -> Result<Self> {
        let types = (
            left.fields().get(spec.left_column).map(|f| f.data_type()),
            right.fields().get(spec.right_column).map(|f| f.data_type()),
        );
        match types {
            (Some(a), Some(b)) if same_layout(a, b) => {}
            _ => {
                return Err(internal(
                    "an as-of join's columns are missing or of two types".to_owned(),
                ));
            }
        }
        Ok(Self {
            left_column: spec.left_column,
            right_column: spec.right_column,
            left: KeyEncoder::new(left, &[SortKey::asc(spec.left_column)])?,
            right: KeyEncoder::new(right, &[SortKey::asc(spec.right_column)])?,
            backward: spec.direction == AsOfMergeDirection::Backward,
            allow_exact: spec.allow_exact,
        })
    }

    /// Whether right value `r` may match left value `l`.
    fn passes(&self, r: &[u8], l: &[u8]) -> bool {
        match (self.backward, self.allow_exact) {
            (true, true) => r <= l,
            (true, false) => r < l,
            (false, true) => r >= l,
            (false, false) => r > l,
        }
    }
}

/// A right row: its batch, the batch's serial and the row.
#[derive(Clone)]
struct Pick {
    batch: Arc<Batch>,
    serial: u64,
    row: usize,
}

/// Left rows `start..end` of the current left batch being matched.
pub(crate) struct AsOfSlice {
    start: usize,
    end: usize,
    /// The next left row without its match (no residual).
    next: usize,
    /// The left rows' as-of values, from `start`.
    values: Rows,
    /// The last right row that passed `next` (backward, no residual).
    candidate: Option<Pick>,
    /// Per left row its best match so far (with a residual).
    best: Vec<Option<Pick>>,
    /// The current chunk's as-of values, from its first row.
    chunk_values: Option<(u64, Rows)>,
}

impl Merge {
    /// The slice of left rows `start..end` of the current batch, its as-of values encoded.
    pub(super) fn as_of_slice(&self, a: &AsOfMerge, start: usize, end: usize) -> Result<AsOfSlice> {
        let l = self
            .left
            .current
            .as_ref()
            .ok_or_else(|| internal("an as-of slice without a left batch".to_owned()))?;
        let values = a.left.encode(&l.batch.data().slice(start, end - start))?;
        let best = if self.shared.filter.is_some() {
            vec![None; end - start]
        } else {
            Vec::new()
        };
        Ok(AsOfSlice {
            start,
            end,
            next: start,
            values,
            candidate: None,
            best,
            chunk_values: None,
        })
    }

    /// One step of a slice: the current chunk of the group, or the slice's end.
    pub(super) async fn as_of_step(&mut self, mut slice: AsOfSlice) -> Result<()> {
        self.query.check_cancelled()?;
        let Some(a) = self.as_of.clone() else {
            return Err(internal("an as-of step without its match".to_owned()));
        };
        let done = self.shared.filter.is_none() && slice.next >= slice.end;
        if done || !self.next_chunk().await? {
            return self.as_of_end(&a, &slice);
        }
        let Some(c) = self.chunks.current.clone() else {
            return self.as_of_end(&a, &slice);
        };
        // The chunk's values, encoded once per slice.
        if slice
            .chunk_values
            .as_ref()
            .is_none_or(|(s, _)| *s != c.serial)
        {
            let rows = a
                .right
                .encode(&c.batch.data().slice(c.start, c.end - c.start))?;
            slice.chunk_values = Some((c.serial, rows));
        }
        if self.shared.filter.is_some() {
            self.as_of_filtered(&a, &mut slice)?;
        } else {
            self.as_of_walk(&a, &mut slice);
        }
        self.phase = Phase::AsOf(slice);
        Ok(())
    }

    /// Without a residual: the current chunk's rows against the slice's next left rows.
    fn as_of_walk(&mut self, a: &AsOfMerge, slice: &mut AsOfSlice) {
        let (Some(c), Some((_, values))) = (self.chunks.current.clone(), slice.chunk_values.take())
        else {
            return;
        };
        let right_nulls = c.batch.data().column(a.right_column).logical_nulls();
        while self.chunks.row < c.end && slice.next < slice.end {
            let r = self.chunks.row;
            if right_nulls.as_ref().is_some_and(|n| n.is_null(r)) {
                self.chunks.row += 1;
                continue;
            }
            let rv = values.row(r - c.start);
            let pick = Pick {
                batch: Arc::clone(&c.batch),
                serial: c.serial,
                row: r,
            };
            // The left rows this right row decides.
            while slice.next < slice.end {
                let i = slice.next;
                if self.left_value_null(a, i) {
                    self.as_of_emit(i, None);
                    slice.next += 1;
                    continue;
                }
                let passes = a.passes(rv.data(), slice.values.row(i - slice.start).data());
                if a.backward {
                    if passes {
                        break;
                    }
                    let candidate = slice.candidate.clone();
                    self.as_of_emit(i, candidate);
                } else {
                    if !passes {
                        break;
                    }
                    self.as_of_emit(i, Some(pick.clone()));
                }
                slice.next += 1;
            }
            if a.backward && slice.next < slice.end {
                slice.candidate = Some(pick);
            }
            self.chunks.row += 1;
        }
        slice.chunk_values = Some((c.serial, values));
    }

    /// With a residual: the slice's rows with up to [`PAIR_BATCH`] right rows' worth of pairs of
    /// the current chunk that the comparison allows, through the residual.
    fn as_of_filtered(&mut self, a: &AsOfMerge, slice: &mut AsOfSlice) -> Result<()> {
        let (Some(l), Some(c), Some(filter), Some((_, values))) = (
            self.left.current.clone(),
            self.chunks.current.clone(),
            self.shared.filter.clone(),
            slice.chunk_values.take(),
        ) else {
            return Ok(());
        };
        let right_nulls = c.batch.data().column(a.right_column).logical_nulls();
        // (left row, chunk row)
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(PAIR_BATCH);
        while self.chunks.row < c.end && pairs.len() < PAIR_BATCH {
            let r = self.chunks.row;
            self.chunks.row += 1;
            if right_nulls.as_ref().is_some_and(|n| n.is_null(r)) {
                continue;
            }
            let rv = values.row(r - c.start);
            for i in slice.start..slice.end {
                let at = i - slice.start;
                // Forward keeps the first passing row: a left row with one is done.
                if (!a.backward && slice.best[at].is_some()) || self.left_value_null(a, i) {
                    continue;
                }
                if a.passes(rv.data(), slice.values.row(at).data()) {
                    pairs.push((i, r));
                }
            }
        }
        slice.chunk_values = Some((c.serial, values));
        if pairs.is_empty() {
            return Ok(());
        }
        let left_idx: Vec<(usize, usize)> = pairs.iter().map(|&(i, _)| (0, i)).collect();
        let right_idx: Vec<(usize, usize)> = pairs.iter().map(|&(_, r)| (0, r)).collect();
        let mut columns = gather(
            std::slice::from_ref(l.batch.data()),
            &left_idx,
            self.left_columns,
        )?;
        columns.extend(gather(
            std::slice::from_ref(c.batch.data()),
            &right_idx,
            self.right_columns,
        )?);
        let candidates = super::record(&self.shared.filter_schema, columns, pairs.len())?;
        let mask = filter.evaluate_in(&candidates, self.partition)?;
        if mask.len() != pairs.len() {
            return Err(internal(
                "a join filter returned the wrong row count".to_owned(),
            ));
        }
        for (j, &(i, r)) in pairs.iter().enumerate() {
            if !mask.is_valid(j) || !mask.value(j) {
                continue;
            }
            let best = &mut slice.best[i - slice.start];
            // Rows come in as-of order: backward keeps the last, forward the first.
            if a.backward || best.is_none() {
                *best = Some(Pick {
                    batch: Arc::clone(&c.batch),
                    serial: c.serial,
                    row: r,
                });
            }
        }
        Ok(())
    }

    /// The group is read for the slice: the left rows still without a match get their last
    /// candidate (backward) or none, and on to the next left rows.
    fn as_of_end(&mut self, a: &AsOfMerge, slice: &AsOfSlice) -> Result<()> {
        if self.shared.filter.is_some() {
            for (at, pick) in slice.best.iter().enumerate() {
                self.as_of_emit(slice.start + at, pick.clone());
            }
        } else {
            for i in slice.next..slice.end {
                let pick = if a.backward && !self.left_value_null(a, i) {
                    slice.candidate.clone()
                } else {
                    None
                };
                self.as_of_emit(i, pick);
            }
        }
        self.left.pos = slice.end;
        self.chunks = super::Chunks::default();
        self.phase = Phase::NextSlice;
        Ok(())
    }

    /// Whether left row `i` of the current batch has a NULL as-of value.
    fn left_value_null(&self, a: &AsOfMerge, i: usize) -> bool {
        self.left
            .current
            .as_ref()
            .is_some_and(|l| l.batch.data().column(a.left_column).is_null(i))
    }

    /// Left row `i` of the current batch with `pick`, or unmatched (a left join pads it).
    fn as_of_emit(&mut self, i: usize, pick: Option<Pick>) {
        let Some(l) = self.left.current.clone() else {
            return;
        };
        match pick {
            Some(p) => {
                let left = self.out.left_source(&l.batch, l.serial);
                let right = self.out.right_source(&p.batch, p.serial);
                self.out.left_rows.push((left, i));
                self.out.right_rows.push((right, p.row));
            }
            None if self.shared.join_type.pads_probe() => {
                let left = self.out.left_source(&l.batch, l.serial);
                self.out.left_rows.push((left, i));
                self.out.right_rows.push((0, 0));
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Fields};

    use super::same_layout;

    fn row(a: &str, b: &str, nullable: bool) -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(a, DataType::Int64, nullable),
            Field::new(b, DataType::Int32, false),
        ]))
    }

    #[test]
    fn values_of_one_layout_compare_whatever_their_field_names() {
        assert!(same_layout(
            &row("req_ts", "seq", false),
            &row("deploy_ts", "seq", true)
        ));
        let list = |f: DataType| DataType::List(Arc::new(Field::new("element", f, true)));
        assert!(same_layout(
            &list(row("a", "b", true)),
            &list(row("c", "d", false))
        ));
        assert!(!same_layout(&row("a", "b", true), &DataType::Int64));
        let wider = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, false),
        ]));
        assert!(!same_layout(&row("a", "b", true), &wider));
        assert!(!same_layout(&DataType::Int32, &DataType::Int64));
    }
}
