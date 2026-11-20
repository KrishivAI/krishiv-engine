// SPDX-License-Identifier: Apache-2.0
//! The Theta sketch aggregates `theta_sketch_agg`, `theta_union_agg` and `theta_intersection_agg` (Spark 4.1+):
//! `ThetaSketchAgg`, `ThetaUnionAgg` and `ThetaIntersectionAgg` at tag v4.3.0 (`thetasketchesAggregates.scala`), over
//! the `datasketches` crate through `krishiv_functions::sketch::theta`.
//!
//! - `theta_sketch_agg` keeps an updatable sketch per group, created with the group. A merge of a partial state
//!   turns the group into a union at the same lgNomEntries (as Spark's `createUnionWith`), which later merges feed.
//! - `theta_union_agg` keeps a union per group; its partial state is the union's compact result.
//! - `theta_intersection_agg` keeps an intersection per group. A group that saw no sketch has no result: its partial
//!   state is the empty byte string (a merge skips it) and its final result is the empty compact sketch.
//!
//! Every group is charged by the library's own `estimated_size` plus a header, recomputed for each group a batch
//! touched (O(1) each). A sketch is trimmed to its nominal size after each batch it was updated by, which keeps the
//! same retained entries as Spark's `rebuild` at the end (the `k` smallest hashes) and releases memory early.
//! NULLs are skipped; so are empty strings, empty binaries and empty arrays. The partial state is the compact,
//! compressed bytes.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float64Array, Int32Array,
    Int64Array, LargeStringArray, ListArray, StringArray, StringViewArray,
};
use krishiv_functions::sketch::theta::{
    self, CompactThetaSketch, DEFAULT_LG_NOM, ThetaIntersection, ThetaSketch, ThetaUnion,
};

use crate::error::{Result, internal};

/// Header and bookkeeping bytes charged on top of a sketch's table.
const OVERHEAD: usize = 128;

/// Bytes charged for a group's fresh state, whatever its kind: a header and a small table.
const NEW_GROUP: usize = OVERHEAD + 512;

/// The most bytes one raw row adds to a sketch that starts empty: a retained hash is 8 bytes, and the table doubles
/// at 15/16 load, so at most about 17 bytes per entry right after a resize; 24 covers it.
const FRESH_BYTES_PER_ROW: usize = 24;

/// Which Theta aggregate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThetaSketchOp {
    /// `theta_sketch_agg` over values, at this lgNomEntries.
    Sketch(u8),
    /// `theta_union_agg` over sketches, at this lgNomEntries.
    Union(u8),
    /// `theta_intersection_agg` over sketches.
    Intersection,
}

/// A group's state.
#[derive(Debug)]
enum Slot {
    /// An updatable sketch (`theta_sketch_agg` before any merge).
    Updatable(ThetaSketch),
    /// A union (`theta_union_agg`, or `theta_sketch_agg` after a merge).
    Union(ThetaUnion),
    /// An intersection (`theta_intersection_agg`).
    Intersection(ThetaIntersection),
}

impl Slot {
    fn charge(&self) -> usize {
        OVERHEAD.saturating_add(match self {
            Self::Updatable(s) => s.estimated_size(),
            Self::Union(u) => u.estimated_size(),
            Self::Intersection(i) => i.estimated_size(),
        })
    }

    /// The compact sketch of the state; `None` for an intersection that saw no sketch.
    fn compact(&self) -> Option<CompactThetaSketch> {
        match self {
            // Trimmed after every batch (`recharge`), so this is Spark's `rebuild.compact`.
            Self::Updatable(s) => Some(s.compact(true)),
            Self::Union(u) => Some(u.to_sketch(true)),
            Self::Intersection(i) => i.to_sketch(true),
        }
    }
}

/// The per-group Theta states of one aggregate.
#[derive(Debug)]
pub(crate) struct ThetaSketchAcc {
    op: ThetaSketchOp,
    slots: Vec<Slot>,
    charges: Vec<usize>,
    total: usize,
    peak: usize,
}

impl ThetaSketchAcc {
    /// A new accumulator; refuses an lgNomEntries outside `4..=26`, as Spark does at the first buffer.
    pub(crate) fn new(op: ThetaSketchOp) -> Result<Self> {
        match op {
            ThetaSketchOp::Sketch(k) => {
                theta::check_lg_nom(i32::from(k), "theta_sketch_agg")?;
            }
            ThetaSketchOp::Union(k) => {
                theta::check_lg_nom(i32::from(k), "theta_union_agg")?;
            }
            ThetaSketchOp::Intersection => {}
        }
        Ok(Self {
            op,
            slots: Vec::new(),
            charges: Vec::new(),
            total: 0,
            peak: 0,
        })
    }

    fn lg_nom(&self) -> u8 {
        match self.op {
            ThetaSketchOp::Sketch(k) | ThetaSketchOp::Union(k) => k,
            ThetaSketchOp::Intersection => u8::try_from(DEFAULT_LG_NOM).unwrap_or(12),
        }
    }

    /// The bytes a state of this aggregate holds at its largest: `2^(lgNom+1)` eight-byte entries.
    fn full(&self) -> usize {
        (8_usize << (u32::from(self.lg_nom()) + 1)).saturating_add(OVERHEAD)
    }

    fn fresh(&self) -> Result<Slot> {
        Ok(match self.op {
            ThetaSketchOp::Sketch(k) => {
                Slot::Updatable(theta::new_sketch(i32::from(k), "theta_sketch_agg")?)
            }
            ThetaSketchOp::Union(k) => {
                Slot::Union(theta::new_union(i32::from(k), "theta_union_agg")?)
            }
            ThetaSketchOp::Intersection => Slot::Intersection(theta::new_intersection()),
        })
    }

    /// Adds groups up to `n` with empty states.
    pub(crate) fn resize(&mut self, n: usize) {
        while self.slots.len() < n {
            let Ok(s) = self.fresh() else { break };
            let c = s.charge();
            self.total = self.total.saturating_add(c);
            self.peak = self.peak.max(c);
            self.charges.push(c);
            self.slots.push(s);
        }
        if n < self.slots.len() {
            self.slots.truncate(n);
            self.charges.truncate(n);
            self.total = self.charges.iter().sum();
            self.peak = self.charges.iter().copied().max().unwrap_or(0);
        }
    }

    /// Room for `new` more groups.
    pub(crate) fn reserve(&mut self, new: usize) {
        self.slots.reserve(new);
        self.charges.reserve(new);
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        self.total.saturating_add(
            self.slots
                .capacity()
                .saturating_mul(std::mem::size_of::<Slot>())
                .saturating_add(
                    self.charges
                        .capacity()
                        .saturating_mul(std::mem::size_of::<usize>()),
                ),
        )
    }

    /// The size after adding `new` groups (an upper bound; a row table passes the partition's rows) and folding
    /// `rows` input rows (`states`: partial states, not raw values).
    ///
    /// Groups that exist grow by at most `groups * ceiling - total charged`; per row, raw values at most double the
    /// table they land in (`2 * peak + header`, capped at the full table), and a state is a whole sketch that can
    /// fill a union's table. Fresh groups cost `NEW_GROUP`, plus `FRESH_BYTES_PER_ROW` a raw row, or a full table a
    /// state. An intersection never holds more than its smallest input; it is bounded by the default lgNomEntries
    /// table here and charged at its real size after each fold.
    pub(crate) fn projected(&self, new: usize, rows: usize, states: bool) -> usize {
        let full = self.full();
        let ceiling = full.max(2 * NEW_GROUP);
        let room = self
            .slots
            .len()
            .saturating_mul(ceiling)
            .saturating_sub(self.total);
        let raw = matches!(self.op, ThetaSketchOp::Sketch(_)) && !states;
        let per_row = if raw {
            self.peak
                .saturating_mul(2)
                .saturating_add(OVERHEAD)
                .min(full)
        } else {
            full
        };
        let landing = rows.saturating_mul(per_row).min(room);
        let fresh = if raw {
            if new > 0 {
                rows.saturating_mul(FRESH_BYTES_PER_ROW)
            } else {
                0
            }
        } else {
            new.saturating_mul(full)
        };
        let transient = if raw { 0 } else { full };
        self.size()
            .saturating_add(new.saturating_mul(NEW_GROUP))
            .saturating_add(landing)
            .saturating_add(fresh)
            .saturating_add(transient)
    }

    fn slot_mut(&mut self, g: usize) -> Result<&mut Slot> {
        self.slots
            .get_mut(g)
            .ok_or_else(|| internal("a Theta group past the accumulator's groups".to_owned()))
    }

    /// Trims the updatable sketches in `groups` and recomputes their charges (O(1) each).
    fn recharge(&mut self, groups: &[u32]) {
        for &g in groups {
            let g = g as usize;
            let (Some(s), Some(c)) = (self.slots.get_mut(g), self.charges.get_mut(g)) else {
                continue;
            };
            if let Slot::Updatable(sketch) = s {
                sketch.trim();
            }
            let now = s.charge();
            self.total = self.total.saturating_sub(*c).saturating_add(now);
            *c = now;
            self.peak = self.peak.max(now);
        }
    }

    /// Folds one batch of raw inputs: values for `theta_sketch_agg`, sketches for the other two. `valid` is false
    /// for a row the `FILTER` mask drops.
    pub(crate) fn update(
        &mut self,
        groups: &[u32],
        arg: &ArrayRef,
        valid: impl Fn(usize) -> bool,
    ) -> Result<()> {
        if arg.len() != groups.len() {
            return Err(internal(
                "a Theta input column of another length than the group ids".to_owned(),
            ));
        }
        match self.op {
            ThetaSketchOp::Sketch(_) => self.update_values(groups, arg, &valid)?,
            ThetaSketchOp::Union(_) | ThetaSketchOp::Intersection => {
                let function = self.function();
                let rows = BinaryRows::of(arg, function)?;
                for (i, &g) in groups.iter().enumerate() {
                    if !valid(i) {
                        continue;
                    }
                    let Some(bytes) = rows.get(i) else { continue };
                    let input = theta::from_bytes(bytes, function)?;
                    self.fold_sketch(g as usize, &input)?;
                }
            }
        }
        self.recharge(groups);
        Ok(())
    }

    fn function(&self) -> &'static str {
        match self.op {
            ThetaSketchOp::Sketch(_) => "theta_sketch_agg",
            ThetaSketchOp::Union(_) => "theta_union_agg",
            ThetaSketchOp::Intersection => "theta_intersection_agg",
        }
    }

    /// Joins `input` into group `g`'s union or intersection (a sketch group becomes a union first).
    fn fold_sketch(&mut self, g: usize, input: &CompactThetaSketch) -> Result<()> {
        let function = self.function();
        let lg = i32::from(self.lg_nom());
        let slot = self.slot_mut(g)?;
        if let Slot::Updatable(s) = slot {
            // Spark's `createUnionWith(buffer.compact, input)`.
            let mut u = theta::new_union(lg, function)?;
            s.trim();
            u.update(&s.compact(true))
                .map_err(|_| internal("a Theta union refused its own sketch".to_owned()))?;
            *slot = Slot::Union(u);
        }
        match slot {
            Slot::Union(u) => u.update(input).map_err(|_| {
                krishiv_common::error::KrishivError::from(
                    krishiv_common::error::ThetaInvalidInputSketchBuffer {
                        function: format!("`{function}`"),
                    },
                )
            })?,
            Slot::Intersection(i) => i.update(input).map_err(|_| {
                krishiv_common::error::KrishivError::from(
                    krishiv_common::error::ThetaInvalidInputSketchBuffer {
                        function: format!("`{function}`"),
                    },
                )
            })?,
            Slot::Updatable(_) => {}
        }
        Ok(())
    }

    fn update_values(
        &mut self,
        groups: &[u32],
        arg: &ArrayRef,
        valid: &impl Fn(usize) -> bool,
    ) -> Result<()> {
        macro_rules! each {
            ($a:expr, $f:expr) => {{
                for (i, &g) in groups.iter().enumerate() {
                    if valid(i) && !$a.is_null(i) {
                        let Slot::Updatable(s) = self.slot_mut(g as usize)? else {
                            return Err(internal(
                                "a Theta union where a sketch was expected".to_owned(),
                            ));
                        };
                        ($f)(s, $a.value(i));
                    }
                }
            }};
        }
        let any = arg.as_any();
        if let Some(a) = any.downcast_ref::<Int32Array>() {
            each!(a, |s: &mut ThetaSketch, v: i32| theta::update_long(
                s,
                i64::from(v)
            ));
        } else if let Some(a) = any.downcast_ref::<Int64Array>() {
            each!(a, |s: &mut ThetaSketch, v: i64| theta::update_long(s, v));
        } else if let Some(a) = any.downcast_ref::<Float64Array>() {
            each!(a, |s: &mut ThetaSketch, v: f64| theta::update_double(s, v));
        } else if let Some(a) = any.downcast_ref::<Float32Array>() {
            each!(a, |s: &mut ThetaSketch, v: f32| theta::update_double(
                s,
                f64::from(v)
            ));
        } else if let Some(a) = any.downcast_ref::<StringArray>() {
            each!(a, |s: &mut ThetaSketch, v: &str| theta::update_str(s, v));
        } else if let Some(a) = any.downcast_ref::<LargeStringArray>() {
            each!(a, |s: &mut ThetaSketch, v: &str| theta::update_str(s, v));
        } else if let Some(a) = any.downcast_ref::<StringViewArray>() {
            each!(a, |s: &mut ThetaSketch, v: &str| theta::update_str(s, v));
        } else if let Some(a) = any.downcast_ref::<BinaryArray>() {
            each!(a, |s: &mut ThetaSketch, v: &[u8]| theta::update_bytes(s, v));
        } else if let Some(a) = any.downcast_ref::<BinaryViewArray>() {
            each!(a, |s: &mut ThetaSketch, v: &[u8]| theta::update_bytes(s, v));
        } else if let Some(a) = any.downcast_ref::<ListArray>() {
            for (i, &g) in groups.iter().enumerate() {
                if !valid(i) || a.is_null(i) {
                    continue;
                }
                let items = a.value(i);
                let Slot::Updatable(s) = self.slot_mut(g as usize)? else {
                    return Err(internal(
                        "a Theta union where a sketch was expected".to_owned(),
                    ));
                };
                if let Some(v) = items.as_any().downcast_ref::<Int32Array>() {
                    // A NULL element reads as 0, as `ArrayData.toIntArray` does.
                    theta::update_ints(s, &v.iter().map(|x| x.unwrap_or(0)).collect::<Vec<_>>());
                } else if let Some(v) = items.as_any().downcast_ref::<Int64Array>() {
                    theta::update_longs(s, &v.iter().map(|x| x.unwrap_or(0)).collect::<Vec<_>>());
                } else {
                    return Err(internal(format!(
                        "theta_sketch_agg over {:?}",
                        arg.data_type()
                    )));
                }
            }
        } else {
            return Err(internal(format!(
                "theta_sketch_agg over {:?}",
                arg.data_type()
            )));
        }
        Ok(())
    }

    /// Merges the partial-state sketches into `groups`. An empty state (an intersection that saw nothing, or a
    /// group with no state) contributes nothing.
    pub(crate) fn merge(&mut self, groups: &[u32], states: &BinaryArray) -> Result<()> {
        if states.len() != groups.len() {
            return Err(internal(
                "a Theta state of another length than its groups".to_owned(),
            ));
        }
        let function = self.function();
        for (row, &g) in groups.iter().enumerate() {
            if states.is_null(row) || states.value(row).is_empty() {
                continue;
            }
            let input = theta::from_bytes(states.value(row), function)?;
            self.fold_sketch(g as usize, &input)?;
        }
        self.recharge(groups);
        Ok(())
    }

    /// The bytes of group `g`'s state; empty for an intersection that saw no sketch.
    fn bytes(&self, g: u32) -> Result<Vec<u8>> {
        let slot = self
            .slots
            .get(g as usize)
            .ok_or_else(|| internal("a Theta group past the accumulator's groups".to_owned()))?;
        Ok(slot
            .compact()
            .map(|c| theta::to_bytes(&c))
            .unwrap_or_default())
    }

    /// The partial state of groups `ids`: the sketch bytes as `binary`.
    pub(crate) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        let rows = ids
            .iter()
            .map(|&g| self.bytes(g))
            .collect::<Result<Vec<_>>>()?;
        Ok(vec![
            Arc::new(BinaryArray::from_iter_values(rows)) as ArrayRef
        ])
    }

    /// The result of groups `ids`, on the `binary` view carrier; an intersection that saw nothing is the empty sketch.
    pub(crate) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        let mut rows = Vec::with_capacity(ids.len());
        for &g in ids {
            let b = self.bytes(g)?;
            rows.push(if b.is_empty() {
                let mut empty = theta::new_sketch(DEFAULT_LG_NOM, self.function())?;
                theta::to_bytes(&theta::compact(&mut empty))
            } else {
                b
            });
        }
        Ok(Arc::new(BinaryViewArray::from_iter_values(rows)) as ArrayRef)
    }
}

/// A binary column on either carrier.
enum BinaryRows<'a> {
    Plain(&'a BinaryArray),
    View(&'a BinaryViewArray),
}

impl<'a> BinaryRows<'a> {
    fn of(arg: &'a ArrayRef, function: &str) -> Result<Self> {
        if let Some(a) = arg.as_any().downcast_ref::<BinaryArray>() {
            Ok(Self::Plain(a))
        } else if let Some(a) = arg.as_any().downcast_ref::<BinaryViewArray>() {
            Ok(Self::View(a))
        } else {
            Err(internal(format!(
                "{function} over {:?}, not binary",
                arg.data_type()
            )))
        }
    }

    /// Row `i`, `None` for NULL.
    fn get(&self, i: usize) -> Option<&[u8]> {
        match self {
            Self::Plain(a) => (!a.is_null(i)).then(|| a.value(i)),
            Self::View(a) => (!a.is_null(i)).then(|| a.value(i)),
        }
    }
}

#[cfg(test)]
mod tests;
