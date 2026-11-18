// SPDX-License-Identifier: Apache-2.0
//! The HLL sketch aggregates `hll_sketch_agg` and `hll_union_agg`: Spark's `HllSketchAgg` and `HllUnionAgg` at tag
//! v4.2.0 (`datasketchesAggregates.scala`), over the `datasketches` crate through `krishiv_functions::sketch::hll`.
//!
//! `hll_sketch_agg` keeps one `HLL_8` sketch per group, created with the group (as Spark's
//! `createAggregationBuffer`). A sketch starts in list mode and grows to set mode and then to `2^lgK` registers, so
//! a group is charged by its coupon count (8 bytes each, a bound for the list and set tables) plus a header, capped
//! at the full `2^lgK`; the charge is recomputed for each group a batch touched. NULLs and empty strings and binaries are skipped. The partial state is the sketch's bytes;
//! merging unions it into the group's sketch. An empty group is the empty sketch, never NULL.
//!
//! `hll_union_agg` keeps a union per group, created on the group's first sketch at that sketch's lgK (an empty
//! group has none: its state is the empty byte string and its result the empty union of the default lgK). Its
//! charge counts a union at its lgK once it exists and at the default one when the groups are added.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Int32Array, Int64Array, LargeStringArray,
    StringArray, StringViewArray,
};
use krishiv_functions::sketch::hll::{self, HllSketch, HllType, HllUnion, UNION_AGG};

use crate::error::{Result, internal};

/// The default lgConfigK (`HllSketch.DEFAULT_LG_K`).
const DEFAULT_LG: u8 = 12;

/// Header and bookkeeping bytes charged on top of a sketch's registers.
const OVERHEAD: usize = 128;

/// Bytes charged for a sketch or union of `lg_k` at its largest.
fn charge(lg_k: u8) -> usize {
    (1_usize << lg_k) + OVERHEAD
}

/// Bytes a list or set table holds per coupon, with its load factor.
const COUPON_BYTES: usize = 8;

/// Bytes charged for `sketch` now: its coupons while it is below the register array, the array after.
fn sketch_charge(sketch: &HllSketch) -> usize {
    let full = charge(sketch.lg_config_k());
    // In list and set mode the estimate is the coupon count; in HLL mode it passes over the registers, but a
    // group at the full charge is not asked again (`recharge`).
    let est = sketch.estimate();
    let coupons = if est.is_finite() && est >= 0.0 {
        // Truncation is the point: a coupon count.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a non-negative estimate"
        )]
        {
            est.ceil() as usize
        }
    } else {
        usize::MAX
    };
    OVERHEAD
        .saturating_add(coupons.saturating_mul(COUPON_BYTES))
        .min(full)
}

/// Which HLL aggregate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HllSketchOp {
    /// `hll_sketch_agg` over values, at this lgK.
    Sketch(u8),
    /// `hll_union_agg` over sketches; whether different lgK values may be mixed.
    Union(bool),
}

#[derive(Debug)]
enum Groups {
    /// The sketches, their current charges and the sum of the charges.
    Sketches(Vec<HllSketch>, Vec<usize>, usize),
    Unions(Vec<Option<HllUnion>>),
}

/// The per-group HLL sketches or unions of one aggregate.
#[derive(Debug)]
pub(crate) struct HllSketchAcc {
    op: HllSketchOp,
    groups: Groups,
}

/// A fresh sketch of `lg_k` (already checked by the planner).
fn fresh(lg_k: u8) -> Result<HllSketch> {
    hll::new_sketch(i32::from(lg_k))
}

impl HllSketchAcc {
    /// A new accumulator; refuses an lgK outside `4..=21` as Spark does at the first buffer.
    pub(crate) fn new(op: HllSketchOp) -> Result<Self> {
        let groups = match op {
            HllSketchOp::Sketch(k) => {
                hll::check_lg_k(i32::from(k))?;
                Groups::Sketches(Vec::new(), Vec::new(), 0)
            }
            HllSketchOp::Union(_) => Groups::Unions(Vec::new()),
        };
        Ok(Self { op, groups })
    }

    /// Adds groups up to `n` with empty states. A sketch that cannot be made (`new` checked the lgK, so none)
    /// leaves its group missing, which the next access reports as an internal error.
    pub(crate) fn resize(&mut self, n: usize) {
        match (&mut self.groups, self.op) {
            (Groups::Sketches(v, charges, total), HllSketchOp::Sketch(k)) => {
                while v.len() < n {
                    let Ok(s) = fresh(k) else { break };
                    let c = sketch_charge(&s);
                    *total = total.saturating_add(c);
                    charges.push(c);
                    v.push(s);
                }
                v.truncate(n);
                charges.truncate(n);
                *total = charges.iter().sum();
            }
            (Groups::Unions(v), _) => v.resize_with(n, || None),
            (Groups::Sketches(..), HllSketchOp::Union(_)) => {}
        }
    }

    /// Room for `new` more groups.
    pub(crate) fn reserve(&mut self, new: usize) {
        match &mut self.groups {
            Groups::Sketches(v, charges, _) => {
                v.reserve(new);
                charges.reserve(new);
            }
            Groups::Unions(v) => v.reserve(new),
        }
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        match &self.groups {
            Groups::Sketches(v, charges, total) => total.saturating_add(
                v.capacity()
                    .saturating_mul(std::mem::size_of::<HllSketch>())
                    .saturating_add(
                        charges
                            .capacity()
                            .saturating_mul(std::mem::size_of::<usize>()),
                    ),
            ),
            Groups::Unions(v) => v
                .iter()
                .flatten()
                .map(|u| charge(u.lg_max_k()))
                .sum::<usize>()
                .saturating_add(
                    v.capacity()
                        .saturating_mul(std::mem::size_of::<Option<HllUnion>>()),
                ),
        }
    }

    /// The size after adding `new` groups and folding `rows` input rows: a new sketch starts at its header and a
    /// value adds at most one coupon, none of it past the full registers.
    pub(crate) fn projected(&self, new: usize, rows: usize) -> usize {
        let per_group = match self.op {
            HllSketchOp::Sketch(k) => {
                let groups = match &self.groups {
                    Groups::Sketches(v, ..) => v.len().saturating_add(new),
                    Groups::Unions(_) => new,
                };
                let growth = rows
                    .saturating_mul(COUPON_BYTES)
                    .min(groups.saturating_mul(charge(k)));
                return self
                    .size()
                    .saturating_add(new.saturating_mul(OVERHEAD))
                    .saturating_add(growth);
            }
            // The union of a group is allocated on its first sketch: counted at the default lgK here, and at its
            // own by `size` once it exists.
            HllSketchOp::Union(_) => charge(DEFAULT_LG),
        };
        self.size().saturating_add(new.saturating_mul(per_group))
    }

    fn sketch_mut(&mut self, g: usize) -> Result<&mut HllSketch> {
        match &mut self.groups {
            Groups::Sketches(v, ..) => v
                .get_mut(g)
                .ok_or_else(|| internal("an HLL group past the accumulator's groups".to_owned())),
            Groups::Unions(_) => Err(internal(
                "an HLL union where a sketch was expected".to_owned(),
            )),
        }
    }

    /// Recomputes the charge of each sketch group in `groups`, which a fold touched. A group already at the full
    /// registers keeps its charge without another pass over them.
    fn recharge(&mut self, groups: &[u32]) {
        let HllSketchOp::Sketch(k) = self.op else {
            return;
        };
        let Groups::Sketches(v, charges, total) = &mut self.groups else {
            return;
        };
        let full = charge(k);
        for &g in groups {
            let g = g as usize;
            let (Some(s), Some(c)) = (v.get(g), charges.get_mut(g)) else {
                continue;
            };
            if *c >= full {
                continue;
            }
            let now = sketch_charge(s);
            *total = total.saturating_sub(*c).saturating_add(now);
            *c = now;
        }
    }

    fn union_slot(&mut self, g: usize) -> Result<&mut Option<HllUnion>> {
        match &mut self.groups {
            Groups::Unions(v) => v
                .get_mut(g)
                .ok_or_else(|| internal("an HLL group past the accumulator's groups".to_owned())),
            Groups::Sketches(..) => Err(internal(
                "an HLL sketch where a union was expected".to_owned(),
            )),
        }
    }

    /// Folds one batch of raw inputs: values for `hll_sketch_agg`, sketches for `hll_union_agg`. `valid` is false
    /// for a row the `FILTER` mask drops.
    pub(crate) fn update(
        &mut self,
        groups: &[u32],
        arg: &ArrayRef,
        valid: impl Fn(usize) -> bool,
    ) -> Result<()> {
        if arg.len() != groups.len() {
            return Err(internal(
                "an HLL input column of another length than the group ids".to_owned(),
            ));
        }
        match self.op {
            HllSketchOp::Sketch(_) => {
                self.update_values(groups, arg, &valid)?;
                self.recharge(groups);
                Ok(())
            }
            HllSketchOp::Union(allow) => {
                let rows = BinaryRows::of(arg)?;
                for (i, &g) in groups.iter().enumerate() {
                    if !valid(i) {
                        continue;
                    }
                    if let Some(bytes) = rows.get(i) {
                        hll::union_agg_update(self.union_slot(g as usize)?, bytes, allow)?;
                    }
                }
                Ok(())
            }
        }
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
                        ($f)(self.sketch_mut(g as usize)?, $a.value(i));
                    }
                }
            }};
        }
        let any = arg.as_any();
        if let Some(a) = any.downcast_ref::<Int32Array>() {
            each!(a, |s: &mut HllSketch, v: i32| hll::update_long(
                s,
                i64::from(v)
            ));
        } else if let Some(a) = any.downcast_ref::<Int64Array>() {
            each!(a, |s: &mut HllSketch, v: i64| hll::update_long(s, v));
        } else if let Some(a) = any.downcast_ref::<StringArray>() {
            each!(a, |s: &mut HllSketch, v: &str| hll::update_str(s, v));
        } else if let Some(a) = any.downcast_ref::<LargeStringArray>() {
            each!(a, |s: &mut HllSketch, v: &str| hll::update_str(s, v));
        } else if let Some(a) = any.downcast_ref::<StringViewArray>() {
            each!(a, |s: &mut HllSketch, v: &str| hll::update_str(s, v));
        } else if let Some(a) = any.downcast_ref::<BinaryArray>() {
            each!(a, |s: &mut HllSketch, v: &[u8]| hll::update_bytes(s, v));
        } else if let Some(a) = any.downcast_ref::<BinaryViewArray>() {
            each!(a, |s: &mut HllSketch, v: &[u8]| hll::update_bytes(s, v));
        } else {
            return Err(internal(format!(
                "hll_sketch_agg over {:?}",
                arg.data_type()
            )));
        }
        Ok(())
    }

    /// Merges the partial-state sketches into `groups`.
    pub(crate) fn merge(&mut self, groups: &[u32], states: &BinaryArray) -> Result<()> {
        if states.len() != groups.len() {
            return Err(internal(
                "an HLL state of another length than its groups".to_owned(),
            ));
        }
        for (row, &g) in groups.iter().enumerate() {
            if states.is_null(row) || states.value(row).is_empty() {
                continue;
            }
            let bytes = states.value(row);
            match self.op {
                HllSketchOp::Sketch(_) => {
                    let input = hll::from_bytes(bytes, "hll_sketch_agg")?;
                    let slot = self.sketch_mut(g as usize)?;
                    // Spark: `Union(sketch.lgK)` over the buffer and the input, written as `HLL_8`.
                    *slot = hll::union(slot, &input, true, "hll_sketch_agg")?;
                }
                HllSketchOp::Union(allow) => {
                    hll::union_agg_update(self.union_slot(g as usize)?, bytes, allow)?;
                }
            }
        }
        self.recharge(groups);
        Ok(())
    }

    /// The bytes of group `g`'s sketch or union.
    fn bytes(&self, g: u32) -> Result<Vec<u8>> {
        let at = g as usize;
        match &self.groups {
            Groups::Sketches(v, ..) => v
                .get(at)
                .map(hll::to_bytes)
                .ok_or_else(|| internal("an HLL group past the accumulator's groups".to_owned())),
            Groups::Unions(v) => match v.get(at) {
                Some(Some(u)) => Ok(hll::to_bytes(&u.to_sketch(HllType::Hll8))),
                Some(None) => Ok(Vec::new()),
                None => Err(internal(
                    "an HLL group past the accumulator's groups".to_owned(),
                )),
            },
        }
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

    /// The result of groups `ids`: the sketches on the `binary` view carrier; an empty union group is the empty
    /// union of the default lgK.
    pub(crate) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        let mut rows = Vec::with_capacity(ids.len());
        for &g in ids {
            let b = self.bytes(g)?;
            rows.push(if b.is_empty() {
                let empty = HllUnion::new(DEFAULT_LG)
                    .map_err(|_| internal("an empty HLL union".to_owned()))?;
                hll::to_bytes(&empty.to_sketch(HllType::Hll8))
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
    fn of(arg: &'a ArrayRef) -> Result<Self> {
        if let Some(a) = arg.as_any().downcast_ref::<BinaryArray>() {
            Ok(Self::Plain(a))
        } else if let Some(a) = arg.as_any().downcast_ref::<BinaryViewArray>() {
            Ok(Self::View(a))
        } else {
            Err(internal(format!(
                "{UNION_AGG} over {:?}, not binary",
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
