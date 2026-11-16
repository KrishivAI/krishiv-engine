// SPDX-License-Identifier: Apache-2.0
//! `system.partition_advice` (P3-T025): the partition advisor's ranked candidates, one row per table and column.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array};

use super::{Sources, long, text};

/// The table's columns, in layout order. Advice reads every principal's queries, so only a `MONITOR` holder
/// sees any rows.
pub(super) fn partition_advice(sources: &Sources) -> Vec<ArrayRef> {
    // Advice reads every principal's queries: only a `MONITOR` holder sees any.
    let advice = match (&sources.history, sources.monitor.sees_all()) {
        (Some(h), true) => crate::partition_advisor::advise_history(h.dir()),
        _ => Vec::new(),
    };
    let rows: Vec<_> = advice
        .iter()
        .flat_map(|t| t.advice.iter().enumerate().map(move |(i, a)| (t, i, a)))
        .collect();
    vec![
        text(rows.iter().map(|(t, ..)| Some(t.table.as_str()))),
        text(rows.iter().map(|(_, _, a)| Some(a.column.as_str()))),
        Arc::new(Int32Array::from_iter(
            rows.iter().map(|(_, i, _)| i32::try_from(i + 1).ok()),
        )),
        long(rows.iter().map(|(_, _, a)| i64::try_from(a.queries).ok())),
        long(rows.iter().map(|(t, ..)| i64::try_from(t.queries).ok())),
        long(rows.iter().map(|(_, _, a)| i64::try_from(a.distinct).ok())),
        Arc::new(arrow_array::Float64Array::from_iter(
            rows.iter().map(|(_, _, a)| Some(a.pruned_fraction)),
        )),
    ]
}
