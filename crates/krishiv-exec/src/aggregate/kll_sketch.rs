// SPDX-License-Identifier: Apache-2.0
//! The KLL quantile sketch aggregates `kll_sketch_agg_{bigint,float,double}` and `kll_merge_agg_{bigint,float,double}`:
//! Spark's `KllSketchAgg*` and `KllMergeAgg*` at tag v4.3.0 (`kllAggregates.scala`), over the `datasketches` crate
//! through `krishiv_functions::sketch::kll`.
//!
//! `kll_sketch_agg_*` keeps one sketch of `k` (default 200) per group, created with the group; NULLs are skipped, a
//! `float` argument of the double aggregate is widened. `kll_merge_agg_*` keeps `None` per group until its first
//! sketch (then a sketch of that sketch's `k`), or a sketch of the given `k` from the start. The partial state is the
//! sketch's bytes (a `None` group writes the empty sketch of the default `k`, as Spark's `serialize`); merging reads
//! empty bytes as no state. An empty group's result is the empty sketch of `k`, never NULL.
//!
//! A sketch holds its items in vectors that grow with them, so a group is charged its header plus 16 bytes an item
//! (the vectors double), recomputed for each group a fold touched.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array,
};
use krishiv_functions::sketch::kll::{DEFAULT_K, Kll, KllType};

use super::hll_sketch::BinaryRows;
use crate::error::{Result, internal};

/// Header and bookkeeping bytes charged for a sketch.
const OVERHEAD: usize = 256;

/// Bytes charged for an item (8 bytes, doubled by the vector's growth).
const ITEM: usize = 16;

/// Bytes a fold of one raw row can add: an item, and a compaction's temporary copy.
const RAW_ROW: usize = 2 * ITEM;

/// Which KLL aggregate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KllSketchOp {
    /// `kll_sketch_agg_*` over values, with this `k`.
    Sketch(KllType, u16),
    /// `kll_merge_agg_*` over sketches, with this `k` if given.
    Merge(KllType, Option<u16>),
}

impl KllSketchOp {
    fn ty(self) -> KllType {
        match self {
            Self::Sketch(t, _) | Self::Merge(t, _) => t,
        }
    }

    /// The Spark function's name, for errors.
    fn name(self) -> &'static str {
        match self {
            Self::Sketch(KllType::Bigint, _) => "kll_sketch_agg_bigint",
            Self::Sketch(KllType::Float, _) => "kll_sketch_agg_float",
            Self::Sketch(KllType::Double, _) => "kll_sketch_agg_double",
            Self::Merge(KllType::Bigint, _) => "kll_merge_agg_bigint",
            Self::Merge(KllType::Float, _) => "kll_merge_agg_float",
            Self::Merge(KllType::Double, _) => "kll_merge_agg_double",
        }
    }

    /// The `k` of the aggregate (`kValue`): the given one or the default.
    fn k(self) -> u16 {
        match self {
            Self::Sketch(_, k) => k,
            Self::Merge(_, k) => k.unwrap_or(DEFAULT_K),
        }
    }

    /// Whether a group starts with a sketch (`createAggregationBuffer`).
    fn starts_full(self) -> bool {
        !matches!(self, Self::Merge(_, None))
    }

    /// Whether the argument holds sketches.
    pub(crate) fn takes_sketches(self) -> bool {
        matches!(self, Self::Merge(..))
    }
}

/// The per-group KLL sketches of one aggregate.
#[derive(Debug)]
pub(crate) struct KllSketchAcc {
    op: KllSketchOp,
    groups: Vec<Option<Kll>>,
    /// Each group's charge, and their sum.
    charges: Vec<usize>,
    total: usize,
}

fn charge(s: &Option<Kll>) -> usize {
    s.as_ref().map_or(0, |s| {
        OVERHEAD.saturating_add(s.retained().saturating_mul(ITEM))
    })
}

impl KllSketchAcc {
    /// A new accumulator.
    pub(crate) fn new(op: KllSketchOp) -> Result<Self> {
        // `k` is checked by the analyzer; a bad one is an internal error at the first group.
        Ok(Self {
            op,
            groups: Vec::new(),
            charges: Vec::new(),
            total: 0,
        })
    }

    pub(crate) fn takes_sketches(&self) -> bool {
        self.op.takes_sketches()
    }

    fn fresh(&self) -> Option<Kll> {
        if self.op.starts_full() {
            // `k` was range-checked when the aggregate was planned, so this cannot fail; a bad one leaves no sketch.
            Kll::new(self.op.ty(), self.op.k(), self.op.name()).ok()
        } else {
            None
        }
    }

    /// Adds groups up to `n` with their starting states.
    pub(crate) fn resize(&mut self, n: usize) {
        while self.groups.len() < n {
            let s = self.fresh();
            let c = charge(&s);
            self.total = self.total.saturating_add(c);
            self.charges.push(c);
            self.groups.push(s);
        }
        if n < self.groups.len() {
            self.groups.truncate(n);
            self.charges.truncate(n);
            self.total = self.charges.iter().sum();
        }
    }

    /// Room for `new` more groups.
    pub(crate) fn reserve(&mut self, new: usize) {
        self.groups.reserve(new);
        self.charges.reserve(new);
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        self.total.saturating_add(
            self.groups
                .capacity()
                .saturating_mul(std::mem::size_of::<Option<Kll>>())
                .saturating_add(
                    self.charges
                        .capacity()
                        .saturating_mul(std::mem::size_of::<usize>()),
                ),
        )
    }

    /// An upper bound of group `g`'s partial state: its serialized size is under its charge.
    pub(crate) fn state_bound(&self, g: u32) -> usize {
        self.charges
            .get(g as usize)
            .copied()
            .unwrap_or(0)
            .max(OVERHEAD)
    }

    /// The size after adding `new` groups and folding `rows` rows; `state_bytes` are the bytes of the sketches
    /// the rows carry (raw sketches of a merge or partial states). A merged sketch keeps no more items than it and
    /// its input hold, and an item is at most 8 serialized bytes.
    pub(crate) fn projected(
        &self,
        new: usize,
        rows: usize,
        states: bool,
        state_bytes: usize,
    ) -> usize {
        let fresh = new.saturating_mul(OVERHEAD.saturating_add(if self.op.starts_full() {
            0
        } else {
            OVERHEAD
        }));
        let growth = if states || self.op.takes_sketches() {
            // Items are at least 4 serialized bytes (floats); each is charged `ITEM`, and a merge holds the input
            // and the union at once.
            state_bytes
                .saturating_mul(ITEM / 4)
                .saturating_mul(2)
                .saturating_add(rows.saturating_mul(OVERHEAD))
        } else {
            rows.saturating_mul(RAW_ROW)
        };
        self.size().saturating_add(fresh).saturating_add(growth)
    }

    fn slot(&mut self, g: u32) -> Result<&mut Option<Kll>> {
        self.groups
            .get_mut(g as usize)
            .ok_or_else(|| internal("a KLL group past the accumulator's groups".to_owned()))
    }

    /// Recomputes the charge of each group in `groups`, which a fold touched.
    fn recharge(&mut self, groups: &[u32]) {
        for &g in groups {
            let g = g as usize;
            let (Some(s), Some(c)) = (self.groups.get(g), self.charges.get_mut(g)) else {
                continue;
            };
            let now = charge(s);
            self.total = self.total.saturating_sub(*c).saturating_add(now);
            *c = now;
        }
    }

    /// Folds one batch of raw inputs: values for `kll_sketch_agg_*`, sketches for `kll_merge_agg_*`. `valid` is
    /// false for a row the `FILTER` mask drops.
    pub(crate) fn update(
        &mut self,
        groups: &[u32],
        arg: &ArrayRef,
        valid: impl Fn(usize) -> bool,
    ) -> Result<()> {
        if arg.len() != groups.len() {
            return Err(internal(
                "a KLL input column of another length than the group ids".to_owned(),
            ));
        }
        match self.op {
            KllSketchOp::Sketch(..) => self.update_values(groups, arg, &valid)?,
            KllSketchOp::Merge(ty, _) => {
                let rows = BinaryRows::of(arg)?;
                let name = self.op.name();
                for (i, &g) in groups.iter().enumerate() {
                    if !valid(i) {
                        continue;
                    }
                    if let Some(bytes) = rows.get(i) {
                        let input = Kll::from_bytes(ty, bytes, name)?;
                        absorb(self.slot(g)?, input, name)?;
                    }
                }
            }
        }
        self.recharge(groups);
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
                        let Some(s) = self.slot(g)?.as_mut() else {
                            return Err(internal("a KLL group without its sketch".to_owned()));
                        };
                        ($f)(s, $a.value(i));
                    }
                }
            }};
        }
        let any = arg.as_any();
        if let Some(a) = any.downcast_ref::<Int8Array>() {
            each!(a, |s: &mut Kll, v: i8| s.update_long(i64::from(v)));
        } else if let Some(a) = any.downcast_ref::<Int16Array>() {
            each!(a, |s: &mut Kll, v: i16| s.update_long(i64::from(v)));
        } else if let Some(a) = any.downcast_ref::<Int32Array>() {
            each!(a, |s: &mut Kll, v: i32| s.update_long(i64::from(v)));
        } else if let Some(a) = any.downcast_ref::<Int64Array>() {
            each!(a, |s: &mut Kll, v: i64| s.update_long(v));
        } else if let Some(a) = any.downcast_ref::<Float32Array>() {
            if self.op.ty() == KllType::Double {
                each!(a, |s: &mut Kll, v: f32| s.update_f64(f64::from(v)));
            } else {
                each!(a, |s: &mut Kll, v: f32| s.update_f32(v));
            }
        } else if let Some(a) = any.downcast_ref::<Float64Array>() {
            each!(a, |s: &mut Kll, v: f64| s.update_f64(v));
        } else {
            return Err(internal(format!(
                "{} over {:?}",
                self.op.name(),
                arg.data_type()
            )));
        }
        Ok(())
    }

    /// Merges the partial-state sketches into `groups`.
    pub(crate) fn merge(&mut self, groups: &[u32], states: &BinaryArray) -> Result<()> {
        if states.len() != groups.len() {
            return Err(internal(
                "a KLL state of another length than its groups".to_owned(),
            ));
        }
        let name = self.op.name();
        for (row, &g) in groups.iter().enumerate() {
            if states.is_null(row) || states.value(row).is_empty() {
                continue;
            }
            let input = Kll::from_bytes(self.op.ty(), states.value(row), name)?;
            absorb(self.slot(g)?, input, name)?;
        }
        self.recharge(groups);
        Ok(())
    }

    /// The bytes of group `g`'s sketch; no sketch is the empty one of the aggregate's `k`.
    fn bytes(&self, g: u32) -> Result<Vec<u8>> {
        match self.groups.get(g as usize) {
            Some(Some(s)) => Ok(s.to_bytes()),
            Some(None) => Ok(Kll::new(self.op.ty(), self.op.k(), self.op.name())?.to_bytes()),
            None => Err(internal(
                "a KLL group past the accumulator's groups".to_owned(),
            )),
        }
    }

    /// The partial state of groups `ids`: the sketch bytes as `binary`.
    pub(crate) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        let rows = ids
            .iter()
            .map(|&g| self.bytes(g))
            .collect::<Result<Vec<_>>>()?;
        Ok(vec![
            Arc::new(super::theta_sketch::binary_array(rows)?) as ArrayRef
        ])
    }

    /// The result of groups `ids` on the `binary` view carrier.
    pub(crate) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        let rows = ids
            .iter()
            .map(|&g| self.bytes(g))
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(BinaryViewArray::from_iter_values(rows)) as ArrayRef)
    }
}

/// Merges `input` into a group's sketch; a group with none takes it.
fn absorb(slot: &mut Option<Kll>, input: Kll, name: &str) -> Result<()> {
    match slot {
        Some(s) => s.merge(&input, name),
        None => {
            // Spark's `newHeapInstance(inputSketch.getK())` merged with the input: the same sketch.
            *slot = Some(input);
            Ok(())
        }
    }
}
