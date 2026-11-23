// SPDX-License-Identifier: Apache-2.0
//! The as-of merge (Spark 4.3's `SortMergeAsOfJoinExec`): within a key's group, each left row
//! takes the nearest right row by the as-of column, among those the residual keeps.
//!
//! Both inputs are sorted on the keys, then the as-of column ascending with NULLs first, so a
//! key's left rows arrive in as-of order and its group (the right rows, buffered or spilled as
//! for any merge) is in as-of order too. A row whose as-of value is NULL never matches: a left
//! one is unmatched, a right one is skipped.
//!
//! Without a residual the group is walked once (two pointers), whatever the number of left
//! batches its key spans: the position, and backward's candidate, carry from one left batch of the
//! key to the next ([`AsOfCursor`]). Backward (`L >= R`, `L > R`) keeps the last right row that
//! passes the next left row and gives it to every left row it passes; forward (`L <= R`, `L < R`)
//! gives each left row the first right row that passes it. Ties follow the sort: backward takes
//! the last of equal right values, forward the first, as Spark's scanner does.
//!
//! With a residual the nearest passing row may be any row the comparison allows, so pairs are
//! evaluated ([`PAIR_BATCH`] at a time) and each left batch of the key reads the group again, as
//! the plain merge's residual pairs do. Forward emits a left row at its first passing pair, as
//! Spark stops at it. Backward evaluates every pair the comparison allows and emits a row once a
//! right row is past its value; the best rows of the rows still open are copied out of each read
//! chunk, reserved ([`Merge::as_of_compact`]). Values compare by their sort rows (canonical floats;
//! structs and arrays in Spark's order). Because pairs go through the residual a batch at a time,
//! forward may evaluate it on right rows past the two Spark's scan stops at: a residual that
//! raises only there (`10 / q.v` with a later `v = 0`) raises here and not in Spark.
//!
//! Spark's distance ([`AsOfDistance`], planned only when it may raise) is evaluated where Spark
//! evaluates it, over a narrow batch of the columns it reads. SQL (Spark's scan): forward only,
//! on a left row's match and its next passing row; a run of left rows matched at one right row
//! checks its first row only (its `R - L` is the largest of the run). Connect (Spark's
//! `RewriteAsOfJoin`, `min_by` over every passing right row): every passing pair, both
//! directions. Without a residual the farthest pair of a key decides (the distance only grows
//! away from the match): backward its last matched left row with the first right row that passed
//! one, forward its first matched left row with the group's last right row (the group is read to
//! its end); with a residual every passing pair is checked. Backward's first passing right row
//! is kept for the whole key, so one read chunk stays alive beside the one being read, as the
//! candidate's does (unreserved when the group is spilled; one chunk, not one per left row).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow_array::{Array, RecordBatch};
use arrow_row::Rows;
use arrow_schema::{DataType, Schema, SchemaRef};
use krishiv_common::error::KrishivError;
use krishiv_types::batch::Batch;

use super::{Chunk, Keyed, Merge, PAIR_BATCH, Phase, gather};
use crate::error::{Result, internal};
use crate::operators::{KeyEncoder, Predicate, SortKey};

/// Which right rows an as-of merge gives a left row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsOfMergeDirection {
    /// The greatest right value at most the left one.
    Backward,
    /// The least right value at least the left one.
    Forward,
}

/// What an as-of merge matches by: the as-of column of each input, of one type.
#[derive(Debug, Clone, PartialEq)]
pub struct AsOfMergeSpec {
    /// The left (probe) input's as-of column.
    pub left_column: usize,
    /// The right (build) input's as-of column.
    pub right_column: usize,
    /// Which right rows match.
    pub direction: AsOfMergeDirection,
    /// Whether an equal value matches.
    pub allow_exact: bool,
    /// Spark's distance (over probe then build columns; the merge checks `isnotnull` of it),
    /// planned only when it may raise: forward `R - L` on each left row's matched pair and its
    /// next passing pair, as Spark's forward scan evaluates its `orderExpression`; with
    /// [`AsOfMergeSpec::distance_on_every_pair`] (forward `R - L`, backward `L - R`) where every
    /// passing pair would raise. It raises what Spark raises there.
    pub distance: Option<crate::physical::Residual>,
    /// Spark's Connect form (`RewriteAsOfJoin`'s `min_by`): the distance of every passing pair
    /// counts, in both directions.
    pub distance_on_every_pair: bool,
}

/// An as-of join's distance check compiled for the merge: Spark's `orderExpression` over a
/// narrow batch of only the columns it reads (`left_columns` of the probe input, then
/// `right_columns` of the build input, as `schema`). The merge evaluates it where Spark
/// evaluates the distance; only what it raises matters.
#[derive(Debug, Clone)]
pub struct AsOfDistance {
    /// The check: `isnotnull(distance)` over `schema`.
    pub check: Arc<dyn Predicate>,
    /// The probe input's columns the distance reads, in `schema` order.
    pub left_columns: Vec<usize>,
    /// The build input's columns the distance reads, after the probe's in `schema`.
    pub right_columns: Vec<usize>,
    /// The narrow batch's schema.
    pub schema: SchemaRef,
}

/// An as-of merge's encoders.
pub(crate) struct AsOfMerge {
    pub(crate) left_column: usize,
    pub(crate) right_column: usize,
    left: Arc<KeyEncoder>,
    right: Arc<KeyEncoder>,
    backward: bool,
    allow_exact: bool,
    /// Spark's distance check, compiled ([`AsOfMergeSpec::distance`]).
    distance: Option<AsOfDistance>,
    /// The distance counts on every passing pair ([`AsOfMergeSpec::distance_on_every_pair`]).
    every_pair: bool,
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
    pub(crate) fn new(
        left: &Schema,
        right: &Schema,
        spec: &AsOfMergeSpec,
        distance: Option<AsOfDistance>,
    ) -> Result<Self> {
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
            // Spark's SQL scan evaluates no distance backward.
            distance: (spec.direction == AsOfMergeDirection::Forward
                || spec.distance_on_every_pair)
                .then_some(distance)
                .flatten(),
            every_pair: spec.distance_on_every_pair,
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

/// Where an as-of join without a residual is in the current group. It is kept across the left
/// batches of the group's key (their values only grow), so the group is read once however many
/// left batches the key spans.
#[derive(Default)]
pub(crate) struct AsOfCursor {
    /// The last right row that passed a left row (backward).
    candidate: Option<Pick>,
    /// The current chunk's as-of values, from its first row, by the chunk's serial.
    chunk_values: Option<(u64, Rows)>,
    /// Every right row of the group was read.
    exhausted: bool,
    /// Forward with a distance (SQL form): the one left row whose next passing right row is still
    /// to come (Spark evaluates the distance there before it stops). One at most: only the first
    /// row of a run matched at one right row waits, so one left batch at most is kept.
    awaiting: Option<Awaiting>,
    /// Pairs (left batch, left row, right row) whose distance is to be evaluated.
    checks: Vec<(Arc<Keyed>, usize, Pick)>,
    /// Every-pair distance without a residual: the left row of the key's farthest passing pair
    /// (backward the last matched left row, forward the first).
    extreme_left: Option<(Arc<Keyed>, usize)>,
    /// Its right row: backward the first right row that passed a left row, forward the group's
    /// last right row with a value (read to the group's end at the key's end).
    extreme_right: Option<Pick>,
}

/// A left row matched at a right row, waiting for the next right row with a value.
struct Awaiting {
    left: Arc<Keyed>,
    row: usize,
    /// The right row it matched, as (chunk serial, row).
    at: (u64, usize),
}

impl AsOfCursor {
    /// The waiting row's check against `pick`, when `pick` is not the row it matched.
    fn take_awaiting(&mut self, pick: &Pick) {
        if self
            .awaiting
            .as_ref()
            .is_some_and(|w| w.at != (pick.serial, pick.row))
            && let Some(w) = self.awaiting.take()
        {
            self.checks.push((w.left, w.row, pick.clone()));
        }
    }
}

/// Left rows `start..end` of the current left batch being matched.
pub(crate) struct AsOfSlice {
    start: usize,
    end: usize,
    /// The first left row without its match: the rows before it are emitted (without a residual,
    /// and backward with one).
    next: usize,
    /// The left rows' as-of values, from `start`.
    values: Rows,
    /// Backward with a residual: per left row from `start`, its best match so far.
    best: Vec<Option<Pick>>,
    /// Forward with a residual: the left rows still without a match, ascending.
    open: Vec<usize>,
    /// Forward with a residual: the left rows before it have a value the right rows read reach.
    reached: usize,
    /// Forward with a residual and a distance: matched left rows whose next passing pair is to
    /// come.
    awaiting: Vec<usize>,
    /// The bytes reserved for the copied best rows (backward with a residual).
    held: usize,
}

impl Merge {
    /// The slice of left rows `start..end` of the current batch, its as-of values encoded. Its
    /// rows with a NULL value (they sort first) are unmatched at once.
    pub(super) fn as_of_slice(
        &mut self,
        a: &AsOfMerge,
        start: usize,
        end: usize,
    ) -> Result<AsOfSlice> {
        let l = self
            .left
            .current
            .clone()
            .ok_or_else(|| internal("an as-of slice without a left batch".to_owned()))?;
        let values = a.left.encode(&l.batch.data().slice(start, end - start))?;
        let nulls = l.batch.data().column(a.left_column).logical_nulls();
        let mut next = start;
        while next < end && nulls.as_ref().is_some_and(|n| n.is_null(next)) {
            self.as_of_emit(next, None);
            next += 1;
        }
        let filtered = self.shared.filter.is_some();
        if filtered {
            // With a residual each slice reads the group from its first row, as the plain
            // merge's residual pairs do: a slice's rows may need right rows the last one passed.
            self.end_chunks();
            self.as_of_cursor.exhausted = false;
        }
        Ok(AsOfSlice {
            start,
            end,
            next,
            values,
            best: if filtered && a.backward {
                vec![None; end - start]
            } else {
                Vec::new()
            },
            open: if filtered && !a.backward {
                (next..end).collect()
            } else {
                Vec::new()
            },
            reached: next,
            awaiting: Vec::new(),
            held: 0,
        })
    }

    /// One step of a slice: the current chunk of the group, or the slice's end.
    pub(super) async fn as_of_step(&mut self, mut slice: AsOfSlice) -> Result<()> {
        self.query.check_cancelled()?;
        let Some(a) = self.as_of.clone() else {
            return Err(internal("an as-of step without its match".to_owned()));
        };
        let filtered = self.shared.filter.is_some();
        let decided = if filtered && !a.backward {
            slice.open.is_empty() && slice.awaiting.is_empty()
        } else {
            slice.next >= slice.end
        };
        if decided {
            return self.as_of_end(&a, slice);
        }
        if self.as_of_cursor.exhausted || !self.next_chunk().await? {
            // No next right row: nothing left to check for the rows that wait for one.
            self.as_of_cursor.exhausted = true;
            self.as_of_cursor.awaiting = None;
            return self.as_of_end(&a, slice);
        }
        let Some(c) = self.chunks.current.clone() else {
            return self.as_of_end(&a, slice);
        };
        // The chunk's values, encoded once.
        let values = match self.as_of_cursor.chunk_values.take() {
            Some((serial, rows)) if serial == c.serial => rows,
            _ => a
                .right
                .encode(&c.batch.data().slice(c.start, c.end - c.start))?,
        };
        let visited = if filtered {
            self.as_of_filtered(&a, &mut slice, &c, &values)
        } else {
            Ok(self.as_of_walk(&a, &mut slice, &c, &values))
        };
        self.as_of_cursor.chunk_values = Some((c.serial, values));
        let visited = visited?;
        self.as_of_checks(&a)?;
        self.counters
            .as_of_right_rows
            .fetch_add(visited, Ordering::Relaxed);
        if filtered && a.backward && self.chunks.row >= c.end {
            self.as_of_compact(&mut slice, c.serial).await?;
            let mut held: Vec<u64> = slice.best[slice.next - slice.start..]
                .iter()
                .flatten()
                .map(|p| p.serial)
                .collect();
            held.sort_unstable();
            held.dedup();
            self.counters
                .as_of_held_chunks
                .fetch_max(held.len() as u64, Ordering::Relaxed);
        }
        self.phase = Phase::AsOf(slice);
        Ok(())
    }

    /// Without a residual: the current chunk's rows against the slice's next left rows, from where
    /// the last slice of the key left the group. The right rows it moved past, as a count.
    fn as_of_walk(
        &mut self,
        a: &AsOfMerge,
        slice: &mut AsOfSlice,
        c: &Chunk,
        values: &Rows,
    ) -> u64 {
        let right_nulls = c.batch.data().column(a.right_column).logical_nulls();
        let mut visited = 0;
        while self.chunks.row < c.end && slice.next < slice.end {
            let r = self.chunks.row;
            if right_nulls.as_ref().is_some_and(|n| n.is_null(r)) {
                self.chunks.row += 1;
                visited += 1;
                continue;
            }
            let pick = || Pick {
                batch: Arc::clone(&c.batch),
                serial: c.serial,
                row: r,
            };
            // The first right row after the waiting row's match passes it too (forward values
            // only grow): its distance is checked there.
            if self.as_of_cursor.awaiting.is_some() {
                self.as_of_cursor.take_awaiting(&pick());
            }
            if a.distance.is_some() && a.every_pair && !a.backward && r + 1 == c.end {
                // Forward's farthest right row so far: a chunk's last row has the largest value
                // of the chunk (NULLs sort first, so it has one when any row has).
                self.as_of_cursor.extreme_right = Some(pick());
            }
            let passes = a.passes(
                values.row(r - c.start).data(),
                slice.values.row(slice.next - slice.start).data(),
            );
            match (a.backward, passes) {
                // The best so far for the next left row and every later one (their values are
                // not smaller).
                (true, true) => {
                    if a.distance.is_some()
                        && a.every_pair
                        && self.as_of_cursor.extreme_right.is_none()
                    {
                        // The smallest right value any left row passes: the farthest.
                        self.as_of_cursor.extreme_right = Some(pick());
                    }
                    self.as_of_cursor.candidate = Some(pick());
                    self.chunks.row += 1;
                    visited += 1;
                }
                // This row and every later one is past the next left row: its match is decided.
                (true, false) => {
                    let candidate = self.as_of_cursor.candidate.clone();
                    self.as_of_emit(slice.next, candidate);
                    slice.next += 1;
                }
                // The first right row the next left row allows; it may serve the rows after it.
                (false, true) => {
                    self.as_of_emit(slice.next, Some(pick()));
                    // Only the first left row of a run matched at one right row waits: its
                    // value is the smallest, so its `R - L` is the largest of the run, here and
                    // at the next row; if it does not raise, none of the others does.
                    if a.distance.is_some()
                        && !a.every_pair
                        && self.as_of_cursor.awaiting.is_none()
                        && let Some(l) = self.left.current.clone()
                    {
                        self.as_of_cursor
                            .checks
                            .push((Arc::clone(&l), slice.next, pick()));
                        self.as_of_cursor.awaiting = Some(Awaiting {
                            left: l,
                            row: slice.next,
                            at: (c.serial, r),
                        });
                    }
                    slice.next += 1;
                }
                (false, false) => {
                    self.chunks.row += 1;
                    visited += 1;
                }
            }
        }
        visited
    }

    /// With a residual: the pairs of the current chunk's rows that the comparison allows, up to
    /// [`PAIR_BATCH`] at once, through the residual. Forward emits a left row at its first passing
    /// pair; backward keeps each open row's last passing pair and emits the row at the first right
    /// row past its value. Backward thus evaluates every pair the comparison allows (Spark scans
    /// from the left row's position until the residual passes); a row already emitted is not
    /// paired again. The right rows it moved past, as a count.
    fn as_of_filtered(
        &mut self,
        a: &AsOfMerge,
        slice: &mut AsOfSlice,
        c: &Chunk,
        values: &Rows,
    ) -> Result<u64> {
        let (Some(l), Some(filter)) = (self.left.current.clone(), self.shared.filter.clone())
        else {
            return Ok(0);
        };
        let right_nulls = c.batch.data().column(a.right_column).logical_nulls();
        let left_value = |i: usize| slice.values.row(i - slice.start);
        let mut visited = 0;
        // (left row, chunk row), in right-row order.
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        while self.chunks.row < c.end {
            let r = self.chunks.row;
            if right_nulls.as_ref().is_some_and(|n| n.is_null(r)) {
                self.chunks.row += 1;
                visited += 1;
                continue;
            }
            let rv = values.row(r - c.start);
            let open: &[usize] = if a.backward {
                if slice.next < slice.end && !a.passes(rv.data(), left_value(slice.next).data()) {
                    if !pairs.is_empty() {
                        // The pairs gathered so far may still change its best row.
                        break;
                    }
                    let best = slice.best[slice.next - slice.start].take();
                    self.as_of_emit(slice.next, best);
                    slice.next += 1;
                    continue;
                }
                if slice.next >= slice.end {
                    break;
                }
                &[]
            } else {
                while slice.reached < slice.end
                    && a.passes(rv.data(), left_value(slice.reached).data())
                {
                    slice.reached += 1;
                }
                let k = slice.open.partition_point(|&i| i < slice.reached);
                &slice.open[..k]
            };
            let n = if a.backward {
                slice.end - slice.next
            } else {
                open.len() + slice.awaiting.len()
            };
            if n > 0 && !pairs.is_empty() && pairs.len() + n > PAIR_BATCH {
                break;
            }
            if a.backward {
                pairs.extend((slice.next..slice.end).map(|i| (i, r)));
            } else {
                pairs.extend(open.iter().map(|&i| (i, r)));
                pairs.extend(slice.awaiting.iter().map(|&i| (i, r)));
            }
            self.chunks.row += 1;
            visited += 1;
            if pairs.len() >= PAIR_BATCH {
                break;
            }
        }
        if pairs.is_empty() {
            return Ok(visited);
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
        let pick = |r: usize| Pick {
            batch: Arc::clone(&c.batch),
            serial: c.serial,
            row: r,
        };
        let every = a.every_pair && a.distance.is_some();
        if a.backward {
            // Rows come in as-of order: the last passing pair is the nearest. Every passing
            // pair's distance counts on the Connect form.
            for (j, &(i, r)) in pairs.iter().enumerate() {
                if mask.is_valid(j) && mask.value(j) {
                    if every {
                        self.as_of_cursor.checks.push((Arc::clone(&l), i, pick(r)));
                    }
                    slice.best[i - slice.start] = Some(pick(r));
                }
            }
        } else {
            // A row's first passing pair is its match. With a distance its next passing pair is
            // checked too, then the row is done; on the Connect form every later passing pair
            // is, so the row waits to the group's end. One state per slice row (pairs come only
            // from open and waiting rows).
            const OPEN: u8 = 0;
            const WAITING: u8 = 1;
            const DONE: u8 = 2;
            let mut state = vec![OPEN; slice.end - slice.start];
            for &w in &slice.awaiting {
                state[w - slice.start] = WAITING;
            }
            let mut matched: Vec<usize> = Vec::new();
            for (j, &(i, r)) in pairs.iter().enumerate() {
                if !(mask.is_valid(j) && mask.value(j)) {
                    continue;
                }
                let at = i - slice.start;
                match state[at] {
                    WAITING => {
                        self.as_of_cursor.checks.push((Arc::clone(&l), i, pick(r)));
                        if !every {
                            state[at] = DONE;
                        }
                    }
                    OPEN => {
                        self.as_of_emit(i, Some(pick(r)));
                        matched.push(i);
                        if a.distance.is_some() {
                            self.as_of_cursor.checks.push((Arc::clone(&l), i, pick(r)));
                            state[at] = WAITING;
                        } else {
                            state[at] = DONE;
                        }
                    }
                    _ => {}
                }
            }
            slice.awaiting = (slice.start..slice.end)
                .filter(|&i| state[i - slice.start] == WAITING)
                .collect();
            matched.sort_unstable();
            slice.open.retain(|i| matched.binary_search(i).is_err());
        }
        Ok(visited)
    }

    /// The key's left rows are done: the rows still waiting for their next passing right row
    /// (forward, without a residual) get it now, read on from where the walk stopped.
    pub(super) async fn as_of_finish(&mut self) -> Result<()> {
        let Some(a) = self.as_of.clone() else {
            return Ok(());
        };
        // Forward on the Connect form: the farthest right row is the group's last one with a
        // value, so the group is read to its end (Spark's `min_by` reads every row too).
        let to_end = a.every_pair
            && a.distance.is_some()
            && !a.backward
            && self.shared.filter.is_none()
            && self.as_of_cursor.extreme_left.is_some();
        let mut visited = 0;
        while (to_end || self.as_of_cursor.awaiting.is_some()) && !self.as_of_cursor.exhausted {
            self.query.check_cancelled()?;
            if !self.next_chunk().await? {
                self.as_of_cursor.exhausted = true;
                break;
            }
            let Some(c) = self.chunks.current.clone() else {
                break;
            };
            let nulls = c.batch.data().column(a.right_column).logical_nulls();
            while self.chunks.row < c.end {
                let r = self.chunks.row;
                if !nulls.as_ref().is_some_and(|n| n.is_null(r)) {
                    let pick = Pick {
                        batch: Arc::clone(&c.batch),
                        serial: c.serial,
                        row: r,
                    };
                    self.as_of_cursor.take_awaiting(&pick);
                    if to_end {
                        if r + 1 == c.end {
                            self.as_of_cursor.extreme_right = Some(pick);
                        }
                    } else if self.as_of_cursor.awaiting.is_none() {
                        break;
                    }
                }
                self.chunks.row += 1;
                visited += 1;
            }
            // The checks gathered here pin their chunks: evaluate them before the next one.
            self.as_of_checks(&a)?;
        }
        self.counters
            .as_of_right_rows
            .fetch_add(visited, Ordering::Relaxed);
        self.as_of_cursor.awaiting = None;
        if let (Some((left, i)), Some(right)) = (
            self.as_of_cursor.extreme_left.take(),
            self.as_of_cursor.extreme_right.take(),
        ) {
            self.as_of_cursor.checks.push((left, i, right));
        }
        self.as_of_checks(&a)
    }

    /// Evaluates Spark's distance on the pairs gathered for it; only what it raises matters.
    fn as_of_checks(&mut self, a: &AsOfMerge) -> Result<()> {
        let checks = std::mem::take(&mut self.as_of_cursor.checks);
        let Some(distance) = &a.distance else {
            return Ok(());
        };
        if checks.is_empty() {
            return Ok(());
        }
        let mut lefts: Vec<(u64, RecordBatch)> = Vec::new();
        let mut rights: Vec<(u64, RecordBatch)> = Vec::new();
        let source = |sources: &mut Vec<(u64, RecordBatch)>, serial: u64, b: &Batch| match sources
            .iter()
            .position(|(s, _)| *s == serial)
        {
            Some(at) => at,
            None => {
                sources.push((serial, b.data().clone()));
                sources.len() - 1
            }
        };
        let mut left_idx = Vec::with_capacity(checks.len());
        let mut right_idx = Vec::with_capacity(checks.len());
        for (l, i, p) in &checks {
            left_idx.push((source(&mut lefts, l.serial, &l.batch), *i));
            right_idx.push((source(&mut rights, p.serial, &p.batch), p.row));
        }
        let lefts: Vec<RecordBatch> = lefts.into_iter().map(|(_, b)| b).collect();
        let rights: Vec<RecordBatch> = rights.into_iter().map(|(_, b)| b).collect();
        // Only the columns the distance reads (its two values, usually).
        let take = |sources: &[RecordBatch], idx: &[(usize, usize)], cols: &[usize]| {
            cols.iter()
                .map(|&c| {
                    let arrays: Vec<&dyn Array> =
                        sources.iter().map(|b| b.column(c).as_ref()).collect();
                    arrow_select::interleave::interleave(&arrays, idx)
                        .map_err(|e| internal(e.to_string()))
                })
                .collect::<Result<Vec<_>>>()
        };
        let mut columns = take(&lefts, &left_idx, &distance.left_columns)?;
        columns.extend(take(&rights, &right_idx, &distance.right_columns)?);
        let pairs = super::record(&distance.schema, columns, checks.len())?;
        distance.check.evaluate_in(&pairs, self.partition)?;
        Ok(())
    }

    /// Backward with a residual, once a chunk is read: the open rows' best rows are copied into one
    /// compact batch, its bytes reserved, so the slice keeps no read chunk alive (a spilled group
    /// would otherwise come back into memory, unreserved, a chunk per open row).
    async fn as_of_compact(&mut self, slice: &mut AsOfSlice, serial: u64) -> Result<()> {
        let from = slice.next - slice.start;
        if !slice.best[from..]
            .iter()
            .any(|p| p.as_ref().is_some_and(|p| p.serial == serial))
        {
            return Ok(());
        }
        let mut sources: Vec<(u64, RecordBatch)> = Vec::new();
        let mut idx: Vec<(usize, usize)> = Vec::new();
        let mut at: Vec<usize> = Vec::new();
        for (k, p) in slice.best.iter().enumerate().skip(from) {
            let Some(p) = p else { continue };
            let s = match sources.iter().position(|(s, _)| *s == p.serial) {
                Some(s) => s,
                None => {
                    sources.push((p.serial, p.batch.data().clone()));
                    sources.len() - 1
                }
            };
            idx.push((s, p.row));
            at.push(k);
        }
        let batches: Vec<RecordBatch> = sources.into_iter().map(|(_, b)| b).collect();
        let Some(first) = batches.first() else {
            return Ok(());
        };
        let schema = first.schema();
        let columns = gather(&batches, &idx, schema.fields().len())?;
        let copy = super::record(&schema, columns, idx.len())?;
        let bytes = copy.get_array_memory_size();
        // A refused `grow` returns before `slice.held` or any pick changes: the old batches stay
        // referenced and the reservation still matches them, so the error leaves no leak or
        // under-count; `copy` is dropped unreserved.
        if bytes > slice.held {
            self.memory
                .grow(bytes - slice.held)
                .await
                .map_err(KrishivError::from)?;
        } else {
            self.memory.shrink(slice.held - bytes);
        }
        slice.held = bytes;
        self.right.serial += 1;
        let batch = Arc::new(Batch::new(copy));
        for (row, k) in at.into_iter().enumerate() {
            slice.best[k] = Some(Pick {
                batch: Arc::clone(&batch),
                serial: self.right.serial,
                row,
            });
        }
        Ok(())
    }

    /// The slice is decided, or the group is read: the left rows still open get their best row
    /// (backward) or none, and on to the next left rows.
    fn as_of_end(&mut self, a: &AsOfMerge, mut slice: AsOfSlice) -> Result<()> {
        let filtered = self.shared.filter.is_some();
        if filtered && !a.backward {
            for i in std::mem::take(&mut slice.open) {
                self.as_of_emit(i, None);
            }
        } else {
            for i in slice.next..slice.end {
                let pick = match (filtered, a.backward) {
                    (true, _) => slice.best[i - slice.start].take(),
                    (false, true) => self.as_of_cursor.candidate.clone(),
                    (false, false) => None,
                };
                self.as_of_emit(i, pick);
            }
        }
        if slice.held > 0 {
            self.memory.shrink(slice.held);
        }
        self.left.pos = slice.end;
        if filtered {
            self.end_chunks();
        }
        self.phase = Phase::NextSlice;
        Ok(())
    }

    /// Left row `i` of the current batch with `pick`, or unmatched (a left join pads it).
    fn as_of_emit(&mut self, i: usize, pick: Option<Pick>) {
        let Some(l) = self.left.current.clone() else {
            return;
        };
        if pick.is_some()
            && self.shared.filter.is_none()
            && let Some(a) = &self.as_of
            && a.every_pair
            && a.distance.is_some()
            && (a.backward || self.as_of_cursor.extreme_left.is_none())
        {
            // The left row of the key's farthest passing pair: backward the largest matched
            // value (the last), forward the smallest (the first).
            self.as_of_cursor.extreme_left = Some((Arc::clone(&l), i));
        }
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
