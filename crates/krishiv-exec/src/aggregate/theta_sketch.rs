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

/// The most bytes one raw row adds to a sketch that starts empty. The crate resizes its table by 8 (`ResizeFactor::X8`)
/// at half load while below nominal size, so right after a resize a retained entry costs up to 16 slots: 128 bytes.
const FRESH_BYTES_PER_ROW: usize = 128;

/// One row can grow a table 8 times (and the old table is alive while it is copied): `9 * table`.
const GROWTH: usize = 9;

/// A state value of `n` bytes holds at most this many table bytes once folded: an entry is at least a byte or so in
/// the compressed form and a table slot costs up to 32 bytes. A union's table is built with the crate's resize by 8,
/// so at worst 1/16 full (128 bytes an entry); the factor holds for real sketches because the transient term in
/// `projected` doubles it. A hand-made compressed buffer with one-bit deltas can reach 16 table bytes per bit in an
/// intersection: the charge is corrected after the fold.
const TABLE_BYTES_PER_STATE_BYTE: usize = 32;

/// The `binary` column of state rows. Arrow's 32-bit offsets overflow past 2 GiB (and `from_iter_values` wraps
/// without a check), which a batch of large sketches can reach.
fn binary_array(rows: Vec<Vec<u8>>) -> Result<BinaryArray> {
    check_total_len(rows.iter().map(Vec::len))?;
    Ok(BinaryArray::from_iter_values(rows))
}

/// The bytes of the sketches in a binary column: the values' own lengths, for `Binary`, `LargeBinary` and
/// `BinaryView` alike, not the views, the offsets or a slice's parent buffer.
pub(crate) fn sketch_bytes(a: &dyn arrow_array::Array) -> usize {
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    match a.data_type() {
        DataType::Binary => {
            let b = a.as_binary::<i32>();
            let o = b.value_offsets();
            o.last()
                .zip(o.first())
                .map_or(0, |(l, f)| usize::try_from(l - f).unwrap_or(0))
        }
        DataType::LargeBinary => {
            let b = a.as_binary::<i64>();
            let o = b.value_offsets();
            o.last()
                .zip(o.first())
                .map_or(0, |(l, f)| usize::try_from(l - f).unwrap_or(0))
        }
        // The low 32 bits of a view are its length.
        DataType::BinaryView => a
            .as_binary_view()
            .views()
            .iter()
            .map(|v| usize::try_from(*v & u128::from(u32::MAX)).unwrap_or(0))
            .sum(),
        _ => a.get_array_memory_size(),
    }
}

/// The error when the lengths sum past what a `binary` column's 32-bit offsets hold (or overflow `usize`).
fn check_total_len(mut lens: impl Iterator<Item = usize>) -> Result<()> {
    let total = lens.try_fold(0_usize, usize::checked_add);
    if total.is_none_or(|n| i32::try_from(n).is_err()) {
        return Err(internal(
            "the Theta partial states of one batch hold more than 2 GiB".to_owned(),
        ));
    }
    Ok(())
}

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
    /// A union (`theta_union_agg`, or `theta_sketch_agg` after a merge), and the raw rows that came after the
    /// merge: they go into `pending`, built on the first such row, so a row is one hash insert, as before the
    /// merge. `pending` is folded into the union when the state is read or another sketch is merged in.
    Union(ThetaUnion, Option<Box<ThetaSketch>>),
    /// An intersection (`theta_intersection_agg`).
    Intersection(ThetaIntersection),
}

impl Slot {
    fn charge(&self) -> usize {
        OVERHEAD.saturating_add(match self {
            Self::Updatable(s) => s.estimated_size(),
            Self::Union(u, p) => u
                .estimated_size()
                .saturating_add(p.as_ref().map_or(0, |p| p.estimated_size())),
            Self::Intersection(i) => i.estimated_size(),
        })
    }

    /// The compact sketch of the state; `None` for an intersection that saw no sketch.
    fn compact(&self, lg: u8) -> Result<Option<CompactThetaSketch>> {
        match self {
            // Trimmed after every batch (`recharge`), so this is Spark's `rebuild.compact`.
            Self::Updatable(s) => theta::finish(s.compact(true), lg).map(Some),
            Self::Union(u, None) => theta::finish(u.to_sketch(true), lg).map(Some),
            Self::Union(u, Some(p)) => {
                // Read-only: the state is the union of the merged part and the pending raw rows.
                let mut both = theta::new_union(i32::from(lg), "theta_sketch_agg")?;
                let invalid = |_| internal("a Theta union refused its own sketch".to_owned());
                both.update(&theta::finish(u.to_sketch(true), lg)?)
                    .map_err(invalid)?;
                both.update(&theta::finish(p.compact(true), lg)?)
                    .map_err(invalid)?;
                theta::finish(both.to_sketch(true), lg).map(Some)
            }
            Self::Intersection(i) => Ok(i.to_sketch(true)),
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
    /// What a fresh group's state is charged (measured on a real one).
    new_group: usize,
    /// A failure to build a group's state in `resize`, raised when the group is first used.
    fault: Option<String>,
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
        let mut acc = Self {
            op,
            slots: Vec::new(),
            charges: Vec::new(),
            total: 0,
            peak: 0,
            new_group: 0,
            fault: None,
        };
        acc.new_group = acc.fresh()?.charge();
        Ok(acc)
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
                Slot::Union(theta::new_union(i32::from(k), "theta_union_agg")?, None)
            }
            ThetaSketchOp::Intersection => Slot::Intersection(theta::new_intersection()),
        })
    }

    /// Adds groups up to `n` with empty states. `new` built a fresh state with these parameters, so a failure here
    /// is an engine fault; it is kept and raised by the next use of a missing group (`slot_mut`), never dropped.
    pub(crate) fn resize(&mut self, n: usize) {
        while self.slots.len() < n {
            let s = match self.fresh() {
                Ok(s) => s,
                Err(e) => {
                    self.fault = Some(e.to_string());
                    break;
                }
            };
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

    /// Whether the argument column holds sketches (a union or an intersection), not values to sketch.
    pub(crate) fn takes_sketches(&self) -> bool {
        !matches!(self.op, ThetaSketchOp::Sketch(_))
    }

    /// The size after adding `new` groups (an upper bound; a row table passes the partition's rows) and folding
    /// `rows` input rows. `states` says the rows are partial states, not raw values, and `state_bytes` is the byte
    /// length of the sketches in the argument column of the whole batch (partial states, or the raw sketches of a
    /// union or an intersection).
    ///
    /// Raw values: one row can grow the table of the group it lands in 8 times (`GROWTH`), up to the full table, and
    /// the groups that exist have `ceiling - charged` bytes of room in all. A fresh group costs what a real fresh
    /// state is charged, plus `FRESH_BYTES_PER_ROW` for each row it may take, up to a full table. States: a folded
    /// sketch adds at most `TABLE_BYTES_PER_STATE_BYTE` times its bytes (a union also at most a full table), and an
    /// intersection's table follows its first input, so it is bounded by the input and not by lgNomEntries.
    pub(crate) fn projected(
        &self,
        new: usize,
        rows: usize,
        states: bool,
        state_bytes: usize,
    ) -> usize {
        let full = self.full();
        let raw = matches!(self.op, ThetaSketchOp::Sketch(_)) && !states;
        let capped = !matches!(self.op, ThetaSketchOp::Intersection);
        let ceiling = full.max(2 * self.new_group).saturating_mul(GROWTH) / 8;
        let table = |bytes: usize| {
            if capped { bytes.min(ceiling) } else { bytes }
        };
        let fresh_floor = new.saturating_mul(self.new_group);
        let room = self
            .slots
            .len()
            .saturating_mul(ceiling)
            .saturating_sub(self.total);
        let (landing, fresh, transient) = if raw {
            let per_row = table(self.peak.saturating_mul(GROWTH).saturating_add(OVERHEAD));
            let landing = rows.saturating_mul(per_row).min(room);
            let fresh = if new > 0 {
                rows.saturating_mul(FRESH_BYTES_PER_ROW)
                    .min(new.saturating_mul(ceiling))
            } else {
                0
            };
            (landing, fresh, 0)
        } else {
            let input = state_bytes
                .saturating_mul(TABLE_BYTES_PER_STATE_BYTE)
                .saturating_add(rows.saturating_mul(OVERHEAD));
            let both = if capped {
                input.min(self.slots.len().saturating_add(new).saturating_mul(ceiling))
            } else {
                input
            };
            (both, 0, input)
        };
        self.size()
            .saturating_add(fresh_floor)
            .saturating_add(landing)
            .saturating_add(fresh)
            .saturating_add(transient)
    }

    /// The error for a group the accumulator does not have: the engine fault that stopped `resize`, if any.
    fn missing_group(&self) -> krishiv_common::error::KrishivError {
        internal(
            self.fault
                .clone()
                .unwrap_or_else(|| "a Theta group past the accumulator's groups".to_owned()),
        )
    }

    fn slot_mut(&mut self, g: usize) -> Result<&mut Slot> {
        if g >= self.slots.len() {
            return Err(self.missing_group());
        }
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
            match s {
                Slot::Updatable(sketch) => sketch.trim(),
                Slot::Union(_, Some(sketch)) => sketch.trim(),
                _ => {}
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
            ThetaSketchOp::Sketch(_) => {
                self.update_values(groups, arg, &valid)?;
            }
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
        let self_lg = self.lg_nom();
        let lg = i32::from(self_lg);
        let slot = self.slot_mut(g)?;
        if let Slot::Updatable(s) = slot {
            // Spark's `createUnionWith(buffer.compact, input)`.
            let mut u = theta::new_union(lg, function)?;
            s.trim();
            u.update(&theta::finish(s.compact(true), self_lg)?)
                .map_err(|_| internal("a Theta union refused its own sketch".to_owned()))?;
            *slot = Slot::Union(u, None);
        }
        // Raw rows taken after an earlier merge join the union before another sketch does.
        if let Slot::Union(u, pending) = slot
            && let Some(mut one) = pending.take()
        {
            one.trim();
            u.update(&theta::finish(one.compact(true), self_lg)?)
                .map_err(|_| internal("a Theta union refused its own sketch".to_owned()))?;
        }
        let invalid = |_| -> krishiv_common::error::KrishivError {
            krishiv_common::error::ThetaInvalidInputSketchBuffer {
                function: format!("`{function}`"),
            }
            .into()
        };
        match slot {
            Slot::Union(u, _) => u.update(input).map_err(invalid)?,
            Slot::Intersection(i) => i.update(input).map_err(invalid)?,
            Slot::Updatable(_) => {}
        }
        Ok(())
    }

    /// Applies a raw update to group `g`. A group whose table was widened holds a union (its sketch was merged
    /// into one), so the update goes into the group's pending sketch, which stays with the group across batches.
    fn apply(&mut self, g: usize, update: impl FnOnce(&mut ThetaSketch)) -> Result<()> {
        let function = self.function();
        let lg_nom = i32::from(self.lg_nom());
        match self.slot_mut(g)? {
            Slot::Updatable(s) => {
                update(s);
                Ok(())
            }
            Slot::Union(_, pending) => {
                if pending.is_none() {
                    *pending = Some(Box::new(theta::new_sketch(lg_nom, function)?));
                }
                if let Some(one) = pending {
                    update(one);
                }
                Ok(())
            }
            Slot::Intersection(_) => {
                Err(internal("a raw update of a Theta intersection".to_owned()))
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
                        self.apply(g as usize, |s| ($f)(s, $a.value(i)))?;
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
                if let Some(v) = items.as_any().downcast_ref::<Int32Array>() {
                    // A NULL element reads as 0, as `ArrayData.toIntArray` does.
                    self.apply(g as usize, |s| {
                        theta::update_int_words(s, v.len(), v.iter().map(|x| x.unwrap_or(0)));
                    })?;
                } else if let Some(v) = items.as_any().downcast_ref::<Int64Array>() {
                    self.apply(g as usize, |s| {
                        theta::update_long_words(s, v.len(), v.iter().map(|x| x.unwrap_or(0)));
                    })?;
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
            .ok_or_else(|| self.missing_group())?;
        Ok(slot
            .compact(self.lg_nom())?
            .map(|c| theta::to_bytes(&c))
            .unwrap_or_default())
    }

    /// The partial state of groups `ids`: the sketch bytes as `binary`.
    pub(crate) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        let rows = ids
            .iter()
            .map(|&g| self.bytes(g))
            .collect::<Result<Vec<_>>>()?;
        Ok(vec![Arc::new(binary_array(rows)?) as ArrayRef])
    }

    /// The result of groups `ids`, on the `binary` view carrier; an intersection that saw nothing is the empty sketch.
    pub(crate) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        let mut rows = Vec::with_capacity(ids.len());
        let mut empty_bytes: Option<Vec<u8>> = None;
        for &g in ids {
            let b = self.bytes(g)?;
            rows.push(if b.is_empty() {
                if empty_bytes.is_none() {
                    let mut empty = theta::new_sketch(DEFAULT_LG_NOM, self.function())?;
                    empty_bytes = Some(theta::to_bytes(&theta::compact(&mut empty)));
                }
                empty_bytes.clone().unwrap_or_default()
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
