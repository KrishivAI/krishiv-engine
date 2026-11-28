// SPDX-License-Identifier: Apache-2.0
//! Kleene `AND` of several conjuncts, run cheapest and most selective first (study 60 (d) part 2,
//! `research/perf-parity/60-planner-rewrites-step0.md`).
//!
//! A chain `c1 AND c2 AND ... AND cn` (nested `AND`s flattened) is one [`Conjunction`]. A conjunct
//! that never raises and is deterministic ("movable") may run in any order among the movable
//! conjuncts next to it: Kleene `AND` does not depend on the order, and such a conjunct already runs
//! on rows an earlier conjunct made false (`AndOr`, `runs_anywhere`). Every other conjunct (one
//! that may raise, or a nondeterministic one) is a barrier: it runs after every conjunct written
//! before it and before every conjunct written after it, over exactly the rows those before it
//! leave open (not FALSE), so what it raises, and the values a stateful call draws, are as today.
//!
//! The order within each run of movable conjuncts is fixed once per instance, after
//! [`MEASURE_ROWS`] rows: by cost class ([`Cost`]), then by the fraction of rows each kept
//! (not FALSE) when run alone over the measured batches (DuckDB's and ClickHouse's adaptive filter
//! order). A movable conjunct runs over the open rows only when they are few: at most a quarter of
//! the batch for a costly one, a sixty-fourth for a string comparison; a cheap one always runs
//! over the whole batch, which costs less than copying the open rows out and back.
//!
//! `compile` builds one for every scalar `AND` chain, not only a `Filter`'s predicate: projections,
//! `CASE` branches, join conditions and an `OR`'s sides too. The barrier rules keep each correct;
//! each instance measures its own first [`MEASURE_ROWS`] rows.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::DataType as ArrowType;

use super::cost::Cost;
use super::{PhysExpr, Value};
use crate::error::Result;
use crate::planner::build::internal;

/// Rows an instance measures its movable conjuncts over before it fixes their order.
const MEASURE_ROWS: u64 = 65_536;

/// One conjunct.
#[derive(Debug)]
pub(super) struct Part {
    pub(super) expr: Arc<dyn PhysExpr>,
    /// Never raises and is deterministic (see the module docs).
    pub(super) movable: bool,
    pub(super) cost: Cost,
}

/// Rows a movable conjunct was measured over alone, and the rows it kept.
#[derive(Debug, Default)]
struct Measure {
    seen: AtomicU64,
    kept: AtomicU64,
}

/// `parts[0] AND parts[1] AND ...`, in an adaptive order (module docs).
#[derive(Debug)]
struct Conjunction {
    parts: Vec<Part>,
    /// The written order.
    written: Vec<usize>,
    measures: Vec<Measure>,
    measured_rows: AtomicU64,
    /// The order once measured.
    order: OnceLock<Vec<usize>>,
}

/// The conjunction of `parts` (at least two), in their written order.
pub(super) fn conjunction(parts: Vec<Part>) -> Arc<dyn PhysExpr> {
    let written = (0..parts.len()).collect();
    let measures = parts.iter().map(|_| Measure::default()).collect();
    Arc::new(Conjunction {
        parts,
        written,
        measures,
        measured_rows: AtomicU64::new(0),
        order: OnceLock::new(),
    })
}

/// Whether a movable conjunct of cost `cost` runs over only the `open` rows of `rows`.
fn over_open_rows(cost: Cost, open: usize, rows: usize) -> bool {
    match cost {
        Cost::Cheap => false,
        Cost::String => open.saturating_mul(64) <= rows,
        Cost::Costly => open.saturating_mul(4) <= rows,
    }
}

impl Conjunction {
    /// The order the measures give: each run of movable conjuncts between barriers sorted by
    /// cost class, then kept fraction (ties in written order); barriers keep their place.
    fn measured_order(&self) -> Vec<usize> {
        // One load of each measure: other drivers may still be adding to them, and a comparator
        // over live values is not a total order (`slice::sort_by` may panic on one).
        let snapshot: Vec<(u64, u64)> = self
            .measures
            .iter()
            .map(|m| {
                (
                    m.kept.load(Ordering::Relaxed),
                    m.seen.load(Ordering::Relaxed),
                )
            })
            .collect();
        order_of(&self.parts, &snapshot)
    }
}

/// Each run of movable `parts` between barriers sorted by cost class, then by kept fraction
/// `kept / seen` from `measures` (unmeasured: 1; ties in written order); barriers keep their place.
pub(super) fn order_of(parts: &[Part], measures: &[(u64, u64)]) -> Vec<usize> {
    let fraction = |i: usize| match measures.get(i) {
        Some(&(kept, seen)) if seen > 0 => (kept, seen),
        _ => (1, 1),
    };
    let mut order = Vec::with_capacity(parts.len());
    let mut run: Vec<usize> = Vec::new();
    let flush = |run: &mut Vec<usize>, order: &mut Vec<usize>| {
        run.sort_by(|&a, &b| {
            let ((ka, sa), (kb, sb)) = (fraction(a), fraction(b));
            parts[a].cost.cmp(&parts[b].cost).then_with(|| {
                (u128::from(ka) * u128::from(sb)).cmp(&(u128::from(kb) * u128::from(sa)))
            })
        });
        order.append(run);
    };
    for (i, p) in parts.iter().enumerate() {
        if p.movable {
            run.push(i);
        } else {
            flush(&mut run, &mut order);
            order.push(i);
        }
    }
    flush(&mut run, &mut order);
    order
}

impl PhysExpr for Conjunction {
    fn data_type(&self) -> &ArrowType {
        &ArrowType::Boolean
    }

    fn nullable(&self) -> bool {
        self.parts.iter().any(|p| p.expr.nullable())
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        let rows = batch.num_rows();
        let fixed = self.order.get();
        let measuring = fixed.is_none();
        // `open`: no conjunct so far is FALSE; `nulled`: one so far is NULL.
        let mut open = BooleanBuffer::new_set(rows);
        let mut nulled = BooleanBuffer::new_unset(rows);
        let mut scalar = rows > 0;
        for &i in fixed.unwrap_or(&self.written) {
            let part = &self.parts[i];
            let open_rows = open.count_set_bits();
            if open_rows == 0 {
                break;
            }
            let whole = open_rows == rows
                || (part.movable && (measuring || !over_open_rows(part.cost, open_rows, rows)));
            let values: ArrayRef = if whole {
                let v = part.expr.eval(batch, partition)?;
                scalar &= v.is_scalar();
                v.into_array(rows)?
            } else {
                scalar = false;
                super::eval_where(part.expr.as_ref(), batch, &open, partition)?
            };
            let values = values
                .as_boolean_opt()
                .ok_or_else(|| internal("AND of a non-boolean"))?;
            // Without a null buffer every value is valid: no NULL to record, FALSE where unset.
            let is_false = match values.nulls() {
                Some(n) => {
                    nulled = &nulled | &(&!n.inner() & &open);
                    n.inner() & &!values.values()
                }
                None => !values.values(),
            };
            if measuring && part.movable && whole {
                let m = &self.measures[i];
                m.seen.fetch_add(count(rows), Ordering::Relaxed);
                m.kept
                    .fetch_add(count(rows - is_false.count_set_bits()), Ordering::Relaxed);
            }
            open = &open & &!&is_false;
        }
        if measuring
            && self
                .measured_rows
                .fetch_add(count(rows), Ordering::Relaxed)
                .saturating_add(count(rows))
                >= MEASURE_ROWS
        {
            // Another driver may set it first; either order is a measured one.
            let _ = self.order.set(self.measured_order());
        }
        // FALSE where a conjunct is FALSE, else NULL where one is NULL, else TRUE.
        let unknown = &open & &nulled;
        let out: ArrayRef = Arc::new(BooleanArray::new(
            &open & &!&nulled,
            Some(NullBuffer::new(!&unknown)),
        ));
        // Every conjunct a literal (or a cast of one): one value for the batch, as `AndOr` gives.
        Ok(if scalar {
            Value::Scalar(out.slice(0, 1))
        } else {
            Value::Array(out)
        })
    }
}

fn count(rows: usize) -> u64 {
    u64::try_from(rows).unwrap_or(u64::MAX)
}
