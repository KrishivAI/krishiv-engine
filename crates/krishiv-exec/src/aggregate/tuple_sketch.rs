// SPDX-License-Identifier: Apache-2.0
//! The Tuple sketch aggregates `tuple_sketch_agg_{double,integer}`, `tuple_union_agg_*` and
//! `tuple_intersection_agg_*` (Spark 4.2+): `TupleSketchAggBase`, `TupleUnionAgg` and `TupleIntersectionAgg` at tag
//! v4.3.0 (`tupleSketchAgg.scala`, `tupleUnionAgg.scala`, `tupleIntersectionAgg.scala`), over the `datasketches` crate
//! through `krishiv_functions::sketch::tuple`.
//!
//! - `tuple_sketch_agg` keeps an updatable sketch per group. A merge of a partial state turns the group into a union
//!   at the same lgNomEntries and mode (Spark's `createUnionWith`); raw rows taken after that go into a pending sketch
//!   that joins the union when the state is read or another sketch is merged.
//! - `tuple_union_agg` keeps a union per group.
//! - `tuple_intersection_agg` keeps an intersection per group; a group that saw no sketch has no result: its partial
//!   state is the empty byte string (a merge skips it) and its final result is the empty sketch.
//!
//! A row whose key or summary is NULL is skipped, as are empty strings, binaries and arrays. A group is charged by the
//! library's `estimated_size` plus a header, recomputed for each group a batch touched; an updatable sketch is trimmed
//! to its nominal size after each batch. The partial state is the compact sketch's bytes.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeStringArray, ListArray, StringArray, StringViewArray,
};
use krishiv_functions::sketch::tuple::{
    self, Mode, Policy, TupleIntersection, TupleSketch, TupleUnion, Value,
};

use super::hll_sketch::BinaryRows;
use crate::error::{Result, internal};

/// Header and bookkeeping bytes charged on top of a sketch's table.
const OVERHEAD: usize = 256;

/// The most bytes one raw row adds to a group's table (a slot, its entry, a resize's old copy).
const RAW_ROW: usize = 256;

/// Table bytes a folded state byte can become (an entry is a few bytes compact and tens in a table).
const TABLE_BYTES_PER_STATE_BYTE: usize = 32;

/// Which Tuple aggregate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TupleSketchOp {
    /// `tuple_sketch_agg_*` over keys and summaries.
    Sketch,
    /// `tuple_union_agg_*` over sketches.
    Union,
    /// `tuple_intersection_agg_*` over sketches.
    Intersection,
}

impl TupleSketchOp {
    fn name(self, integer: bool) -> &'static str {
        match (self, integer) {
            (Self::Sketch, false) => "tuple_sketch_agg_double",
            (Self::Sketch, true) => "tuple_sketch_agg_integer",
            (Self::Union, false) => "tuple_union_agg_double",
            (Self::Union, true) => "tuple_union_agg_integer",
            (Self::Intersection, false) => "tuple_intersection_agg_double",
            (Self::Intersection, true) => "tuple_intersection_agg_integer",
        }
    }
}

type Sketch<V> = TupleSketch<Policy<V>>;
type Union<V> = TupleUnion<Policy<V>>;
type Intersection<V> = TupleIntersection<Policy<V>>;
type Compact<V> = tuple::Compact<V>;

/// A group's state.
enum Slot<V: Value> {
    Updatable(Sketch<V>),
    /// A union, and the raw rows taken after a merge.
    Union(Union<V>, Option<Box<Sketch<V>>>),
    Intersection(Intersection<V>),
}

impl<V: Value> std::fmt::Debug for Slot<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Updatable(_) => "Updatable",
            Self::Union(..) => "Union",
            Self::Intersection(_) => "Intersection",
        })
    }
}

impl<V: Value> Slot<V> {
    fn charge(&self) -> usize {
        OVERHEAD.saturating_add(match self {
            Self::Updatable(s) => s.estimated_size(),
            Self::Union(u, p) => u
                .estimated_size()
                .saturating_add(p.as_ref().map_or(0, |p| p.estimated_size())),
            Self::Intersection(i) => i.estimated_size(),
        })
    }
}

/// The per-group states of one aggregate over summaries `V`.
#[derive(Debug)]
pub(crate) struct Acc<V: Value> {
    op: TupleSketchOp,
    integer: bool,
    lg: u8,
    mode: Mode,
    slots: Vec<Slot<V>>,
    charges: Vec<usize>,
    total: usize,
    peak: usize,
    new_group: usize,
    fault: Option<String>,
}

fn invalid(function: &str) -> krishiv_common::error::KrishivError {
    krishiv_common::error::TupleInvalidInputSketchBuffer {
        function: format!("`{function}`"),
    }
    .into()
}

impl<V: Value> Acc<V> {
    fn new(op: TupleSketchOp, integer: bool, lg: u8, mode: Mode) -> Result<Self> {
        let mut acc = Self {
            op,
            integer,
            lg,
            mode,
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

    fn function(&self) -> &'static str {
        self.op.name(self.integer)
    }

    /// The bytes a state holds at its largest: `2^(lg+1)` entries of a hash, a summary and a slot.
    fn full(&self) -> usize {
        let entry = 8 + V::SIZE + 1 + 16;
        (entry << (u32::from(self.lg) + 1)).saturating_add(OVERHEAD)
    }

    fn fresh(&self) -> Result<Slot<V>> {
        let f = self.function();
        let lg = i32::from(self.lg);
        Ok(match self.op {
            TupleSketchOp::Sketch => Slot::Updatable(tuple::new_sketch::<V>(lg, self.mode, f)?),
            TupleSketchOp::Union => Slot::Union(tuple::new_union::<V>(lg, self.mode, f)?, None),
            TupleSketchOp::Intersection => {
                Slot::Intersection(tuple::new_intersection::<V>(self.mode))
            }
        })
    }

    fn takes_sketches(&self) -> bool {
        self.op != TupleSketchOp::Sketch
    }

    fn resize(&mut self, n: usize) {
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

    fn reserve(&mut self, new: usize) {
        self.slots.reserve(new);
        self.charges.reserve(new);
    }

    fn size(&self) -> usize {
        self.total.saturating_add(
            self.slots
                .capacity()
                .saturating_mul(std::mem::size_of::<Slot<V>>())
                .saturating_add(
                    self.charges
                        .capacity()
                        .saturating_mul(std::mem::size_of::<usize>()),
                ),
        )
    }

    fn state_bound(&self, g: u32) -> usize {
        let at = usize::try_from(g).unwrap_or(usize::MAX);
        let charge = self.charges.get(at).copied().unwrap_or(0);
        let scratch = matches!(self.slots.get(at), Some(Slot::Union(_, Some(_))));
        charge
            .saturating_mul(if scratch { 3 } else { 1 })
            .saturating_add(OVERHEAD)
    }

    fn projected(&self, new: usize, rows: usize, states: bool, state_bytes: usize) -> usize {
        let full = self.full();
        let cap = self.slots.len().saturating_add(new).saturating_mul(full);
        let growth = if states || self.takes_sketches() {
            state_bytes
                .saturating_mul(TABLE_BYTES_PER_STATE_BYTE)
                .saturating_add(rows.saturating_mul(OVERHEAD))
        } else {
            rows.saturating_mul(RAW_ROW)
        };
        self.size()
            .saturating_add(new.saturating_mul(self.new_group))
            .saturating_add(growth.min(cap))
            .saturating_add(full.min(growth))
    }

    fn slot_mut(&mut self, g: usize) -> Result<&mut Slot<V>> {
        if g >= self.slots.len() {
            return Err(internal(self.fault.clone().unwrap_or_else(|| {
                "a Tuple group past the accumulator's groups".to_owned()
            })));
        }
        self.slots
            .get_mut(g)
            .ok_or_else(|| internal("a Tuple group past the accumulator's groups".to_owned()))
    }

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

    /// Joins `input` into group `g`'s union or intersection (a sketch group becomes a union first).
    fn fold_sketch(&mut self, g: usize, input: &Compact<V>) -> Result<()> {
        let function = self.function();
        let (lg, mode) = (i32::from(self.lg), self.mode);
        let slot = self.slot_mut(g)?;
        if let Slot::Updatable(s) = slot {
            let mut u = tuple::new_union::<V>(lg, mode, function)?;
            s.trim();
            u.update(&tuple::compact(s))
                .map_err(|_| invalid(function))?;
            *slot = Slot::Union(u, None);
        }
        if let Slot::Union(u, pending) = slot
            && let Some(mut one) = pending.take()
        {
            one.trim();
            u.update(&tuple::compact(&one))
                .map_err(|_| invalid(function))?;
        }
        match slot {
            Slot::Union(u, _) => u.update(input).map_err(|_| invalid(function))?,
            Slot::Intersection(i) => i.update(input).map_err(|_| invalid(function))?,
            Slot::Updatable(_) => {}
        }
        Ok(())
    }

    /// A raw update of group `g`'s sketch (or its pending one after a merge).
    fn apply(&mut self, g: usize, update: impl FnOnce(&mut Sketch<V>)) -> Result<()> {
        let function = self.function();
        let (lg, mode) = (i32::from(self.lg), self.mode);
        match self.slot_mut(g)? {
            Slot::Updatable(s) => update(s),
            Slot::Union(_, pending) => {
                if pending.is_none() {
                    *pending = Some(Box::new(tuple::new_sketch::<V>(lg, mode, function)?));
                }
                if let Some(one) = pending {
                    update(one);
                }
            }
            Slot::Intersection(_) => {
                return Err(internal("a raw update of a Tuple intersection".to_owned()));
            }
        }
        Ok(())
    }

    fn update_sketches(
        &mut self,
        groups: &[u32],
        arg: &ArrayRef,
        valid: &impl Fn(usize) -> bool,
    ) -> Result<()> {
        let function = self.function();
        let rows = BinaryRows::of(arg)?;
        for (i, &g) in groups.iter().enumerate() {
            if !valid(i) {
                continue;
            }
            let Some(bytes) = rows.get(i) else { continue };
            let input = tuple::from_bytes::<V>(bytes, function)?;
            self.fold_sketch(g as usize, &input)?;
        }
        Ok(())
    }

    fn update_keys(
        &mut self,
        groups: &[u32],
        key: &ArrayRef,
        summary: &dyn Fn(usize) -> Option<V>,
        valid: &impl Fn(usize) -> bool,
    ) -> Result<()> {
        macro_rules! each {
            ($a:expr, $f:expr) => {{
                for (i, &g) in groups.iter().enumerate() {
                    if valid(i) && !$a.is_null(i) {
                        let Some(v) = summary(i) else { continue };
                        self.apply(g as usize, |s| ($f)(s, $a.value(i), v))?;
                    }
                }
            }};
        }
        let any = key.as_any();
        if let Some(a) = any.downcast_ref::<Int32Array>() {
            each!(a, |s: &mut Sketch<V>, k: i32, v| tuple::update_long(
                s,
                i64::from(k),
                v
            ));
        } else if let Some(a) = any.downcast_ref::<Int64Array>() {
            each!(a, |s: &mut Sketch<V>, k: i64, v| tuple::update_long(
                s, k, v
            ));
        } else if let Some(a) = any.downcast_ref::<Float64Array>() {
            each!(a, |s: &mut Sketch<V>, k: f64, v| tuple::update_double(
                s, k, v
            ));
        } else if let Some(a) = any.downcast_ref::<Float32Array>() {
            each!(a, |s: &mut Sketch<V>, k: f32, v| tuple::update_double(
                s,
                f64::from(k),
                v
            ));
        } else if let Some(a) = any.downcast_ref::<StringArray>() {
            each!(a, |s: &mut Sketch<V>, k: &str, v| tuple::update_bytes(
                s,
                k.as_bytes(),
                v
            ));
        } else if let Some(a) = any.downcast_ref::<LargeStringArray>() {
            each!(a, |s: &mut Sketch<V>, k: &str, v| tuple::update_bytes(
                s,
                k.as_bytes(),
                v
            ));
        } else if let Some(a) = any.downcast_ref::<StringViewArray>() {
            each!(a, |s: &mut Sketch<V>, k: &str, v| tuple::update_bytes(
                s,
                k.as_bytes(),
                v
            ));
        } else if let Some(a) = any.downcast_ref::<BinaryArray>() {
            each!(a, |s: &mut Sketch<V>, k: &[u8], v| tuple::update_bytes(
                s, k, v
            ));
        } else if let Some(a) = any.downcast_ref::<BinaryViewArray>() {
            each!(a, |s: &mut Sketch<V>, k: &[u8], v| tuple::update_bytes(
                s, k, v
            ));
        } else if let Some(a) = any.downcast_ref::<ListArray>() {
            for (i, &g) in groups.iter().enumerate() {
                if !valid(i) || a.is_null(i) {
                    continue;
                }
                let Some(v) = summary(i) else { continue };
                let items = a.value(i);
                if let Some(w) = items.as_any().downcast_ref::<Int32Array>() {
                    // A NULL element reads as 0, as `ArrayData.toIntArray` does.
                    self.apply(g as usize, |s| {
                        tuple::update_int_words(s, w.len(), w.iter().map(|x| x.unwrap_or(0)), v);
                    })?;
                } else if let Some(w) = items.as_any().downcast_ref::<Int64Array>() {
                    self.apply(g as usize, |s| {
                        tuple::update_long_words(s, w.len(), w.iter().map(|x| x.unwrap_or(0)), v);
                    })?;
                } else {
                    return Err(internal(format!(
                        "{} over {:?}",
                        self.function(),
                        key.data_type()
                    )));
                }
            }
        } else {
            return Err(internal(format!(
                "{} over {:?}",
                self.function(),
                key.data_type()
            )));
        }
        Ok(())
    }

    fn merge(&mut self, groups: &[u32], states: &BinaryArray) -> Result<()> {
        if states.len() != groups.len() {
            return Err(internal(
                "a Tuple state of another length than its groups".to_owned(),
            ));
        }
        let function = self.function();
        for (row, &g) in groups.iter().enumerate() {
            if states.is_null(row) || states.value(row).is_empty() {
                continue;
            }
            let input = tuple::from_bytes::<V>(states.value(row), function)?;
            self.fold_sketch(g as usize, &input)?;
        }
        self.recharge(groups);
        Ok(())
    }

    /// The compact sketch of group `g`; `None` for an intersection that saw no sketch.
    fn compact(&self, g: u32) -> Result<Option<Compact<V>>> {
        let function = self.function();
        let slot = self
            .slots
            .get(g as usize)
            .ok_or_else(|| internal("a Tuple group past the accumulator's groups".to_owned()))?;
        Ok(match slot {
            Slot::Updatable(s) => Some(tuple::compact(s)),
            Slot::Union(u, None) => Some(u.to_sketch(false)),
            Slot::Union(u, Some(p)) => {
                let mut both = tuple::new_union::<V>(i32::from(self.lg), self.mode, function)?;
                both.update(&u.to_sketch(false))
                    .map_err(|_| invalid(function))?;
                both.update(&tuple::compact(p))
                    .map_err(|_| invalid(function))?;
                Some(both.to_sketch(false))
            }
            Slot::Intersection(i) => i.to_sketch(false),
        })
    }

    fn bytes(&self, g: u32) -> Result<Vec<u8>> {
        Ok(self
            .compact(g)?
            .map(|c| tuple::to_bytes_at(&c, i32::from(self.lg)))
            .unwrap_or_default())
    }

    fn state_rows(&self, ids: &[u32]) -> Result<Vec<Vec<u8>>> {
        ids.iter().map(|&g| self.bytes(g)).collect()
    }

    fn evaluate_rows(&self, ids: &[u32]) -> Result<Vec<Vec<u8>>> {
        let mut empty: Option<Vec<u8>> = None;
        let mut rows = Vec::with_capacity(ids.len());
        for &g in ids {
            let b = self.bytes(g)?;
            rows.push(if b.is_empty() {
                if empty.is_none() {
                    let s = tuple::new_sketch::<V>(i32::from(self.lg), self.mode, self.function())?;
                    empty = Some(tuple::to_bytes(&tuple::compact(&s)));
                }
                empty.clone().unwrap_or_default()
            } else {
                b
            });
        }
        Ok(rows)
    }
}

/// The summary column of `tuple_sketch_agg_double`, read as `f64`.
fn double_summaries(arg: &ArrayRef) -> Result<Box<dyn Fn(usize) -> Option<f64> + '_>> {
    let any = arg.as_any();
    if let Some(a) = any.downcast_ref::<Float64Array>() {
        Ok(Box::new(move |i| (!a.is_null(i)).then(|| a.value(i))))
    } else if let Some(a) = any.downcast_ref::<Float32Array>() {
        Ok(Box::new(move |i| {
            (!a.is_null(i)).then(|| f64::from(a.value(i)))
        }))
    } else if let Some(a) = any.downcast_ref::<Int32Array>() {
        Ok(Box::new(move |i| {
            (!a.is_null(i)).then(|| f64::from(a.value(i)))
        }))
    } else {
        Err(internal(format!(
            "a Tuple summary over {:?}",
            arg.data_type()
        )))
    }
}

/// The summary column of `tuple_sketch_agg_integer`, read as `i32`.
fn integer_summaries(arg: &ArrayRef) -> Result<Box<dyn Fn(usize) -> Option<i32> + '_>> {
    let any = arg.as_any();
    if let Some(a) = any.downcast_ref::<Int32Array>() {
        Ok(Box::new(move |i| (!a.is_null(i)).then(|| a.value(i))))
    } else if let Some(a) = any.downcast_ref::<Int16Array>() {
        Ok(Box::new(move |i| {
            (!a.is_null(i)).then(|| i32::from(a.value(i)))
        }))
    } else if let Some(a) = any.downcast_ref::<Int8Array>() {
        Ok(Box::new(move |i| {
            (!a.is_null(i)).then(|| i32::from(a.value(i)))
        }))
    } else {
        Err(internal(format!(
            "a Tuple summary over {:?}",
            arg.data_type()
        )))
    }
}

/// The per-group Tuple states of one aggregate (`_double` or `_integer` summaries).
#[derive(Debug)]
pub(crate) enum TupleSketchAcc {
    Double(Acc<f64>),
    Integer(Acc<i32>),
}

macro_rules! each_acc {
    ($self:expr, $a:ident => $e:expr) => {
        match $self {
            Self::Double($a) => $e,
            Self::Integer($a) => $e,
        }
    };
}

impl TupleSketchAcc {
    /// A new accumulator; `lg_nom` outside `4..=26` is refused as Spark does at the first buffer.
    pub(crate) fn new(op: TupleSketchOp, integer: bool, lg_nom: u8, mode: Mode) -> Result<Self> {
        krishiv_functions::sketch::theta::check_lg_nom(i32::from(lg_nom), op.name(integer))?;
        Ok(if integer {
            Self::Integer(Acc::new(op, true, lg_nom, mode)?)
        } else {
            Self::Double(Acc::new(op, false, lg_nom, mode)?)
        })
    }

    pub(crate) fn takes_sketches(&self) -> bool {
        each_acc!(self, a => a.takes_sketches())
    }

    pub(crate) fn resize(&mut self, n: usize) {
        each_acc!(self, a => a.resize(n));
    }

    pub(crate) fn reserve(&mut self, new: usize) {
        each_acc!(self, a => a.reserve(new));
    }

    pub(crate) fn size(&self) -> usize {
        each_acc!(self, a => a.size())
    }

    pub(crate) fn state_bound(&self, g: u32) -> usize {
        each_acc!(self, a => a.state_bound(g))
    }

    pub(crate) fn projected(
        &self,
        new: usize,
        rows: usize,
        states: bool,
        state_bytes: usize,
    ) -> usize {
        each_acc!(self, a => a.projected(new, rows, states, state_bytes))
    }

    /// Folds one batch of raw inputs: `args` is the key and the summary for `tuple_sketch_agg_*`, the sketch column
    /// for the other two. `valid` is false for a row the `FILTER` mask drops or whose first argument is NULL.
    pub(crate) fn update(
        &mut self,
        groups: &[u32],
        args: &[&ArrayRef],
        valid: impl Fn(usize) -> bool,
    ) -> Result<()> {
        if args.iter().any(|a| a.len() != groups.len()) {
            return Err(internal(
                "a Tuple input column of another length than the group ids".to_owned(),
            ));
        }
        match (self, args) {
            (Self::Double(a), [key, summary]) if !a.takes_sketches() => {
                let s = double_summaries(summary)?;
                a.update_keys(groups, key, &*s, &valid)?;
                a.recharge(groups);
            }
            (Self::Integer(a), [key, summary]) if !a.takes_sketches() => {
                let s = integer_summaries(summary)?;
                a.update_keys(groups, key, &*s, &valid)?;
                a.recharge(groups);
            }
            (Self::Double(a), [sketches]) if a.takes_sketches() => {
                a.update_sketches(groups, sketches, &valid)?;
                a.recharge(groups);
            }
            (Self::Integer(a), [sketches]) if a.takes_sketches() => {
                a.update_sketches(groups, sketches, &valid)?;
                a.recharge(groups);
            }
            _ => {
                return Err(internal(
                    "a Tuple aggregate got other argument columns than it reads".to_owned(),
                ));
            }
        }
        Ok(())
    }

    /// Merges the partial-state sketches into `groups`.
    pub(crate) fn merge(&mut self, groups: &[u32], states: &BinaryArray) -> Result<()> {
        each_acc!(self, a => a.merge(groups, states))
    }

    /// The partial state of groups `ids`: the sketch bytes as `binary`.
    pub(crate) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        let rows = each_acc!(self, a => a.state_rows(ids))?;
        Ok(vec![
            Arc::new(super::theta_sketch::binary_array(rows)?) as ArrayRef
        ])
    }

    /// The result of groups `ids`, on the `binary` view carrier.
    pub(crate) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        let rows = each_acc!(self, a => a.evaluate_rows(ids))?;
        Ok(Arc::new(BinaryViewArray::from_iter_values(rows)) as ArrayRef)
    }
}

#[cfg(test)]
mod tests;
