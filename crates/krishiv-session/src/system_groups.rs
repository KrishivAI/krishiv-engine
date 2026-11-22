// SPDX-License-Identifier: Apache-2.0
//! `system.resource_groups` (P3-T066): one row per resource group with its limits and live counters.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, TimestampMicrosecondArray};

use crate::system::{Kind, Sources, long, text};

/// The rows a caller sees. With `Monitor` on `system`: every group. Without it (design §5.4):
/// the groups holding one of the caller's running or queued queries, and their ancestors,
/// whose subtree counters are totals over other tenants' work and show as null; the caller's own
/// group keeps them.
fn visible_rows(sources: &Sources) -> Vec<(crate::resource_groups::GroupRow, bool)> {
    let Some(groups) = &sources.groups else {
        return Vec::new();
    };
    let rows = groups.rows();
    if sources.monitor.sees_all() {
        return rows.into_iter().map(|r| (r, true)).collect();
    }
    let own: Vec<String> = sources
        .queries
        .snapshot(false)
        .into_iter()
        .filter(|q| holds_a_slot_or_a_place(q.state))
        .filter_map(|q| q.resource_group)
        .collect();
    visible_to_owner(rows, &own, |r| r.group_id.as_str())
}

/// Whether a query in `state` counts for the own-groups rule: running, or queued in its group.
pub(crate) fn holds_a_slot_or_a_place(state: &str) -> bool {
    matches!(state, "running" | "queued")
}

/// The one visibility rule of a caller without `Monitor` (design §5.4), shared by
/// `system.resource_groups` and `ListResourceGroups`: the groups in `own` (the groups of the
/// caller's running or queued queries) and their ancestors, each paired with whether it is one of
/// `own` (its counters are shown) or only an ancestor (subtree counters are other tenants' work).
pub(crate) fn visible_to_owner<R>(
    rows: Vec<R>,
    own: &[String],
    id: impl Fn(&R) -> &str,
) -> Vec<(R, bool)> {
    rows.into_iter()
        .filter_map(|r| {
            let holds = own.iter().any(|g| g == id(&r));
            let ancestor = own.iter().any(|g| {
                g.strip_prefix(id(&r))
                    .is_some_and(|rest| rest.starts_with('.'))
            });
            (holds || ancestor).then_some((r, holds))
        })
        .collect()
}

/// The rows of `system.resource_groups` as columns.
pub(crate) fn columns(sources: &Sources) -> Vec<ArrayRef> {
    let visible = visible_rows(sources);
    // The subtree counters are shown for every row when the caller sees all, else only on the
    // caller's own groups.
    let counters: Vec<bool> = visible.iter().map(|(_, own)| *own).collect();
    let rows: Vec<_> = visible.into_iter().map(|(r, _)| r).collect();
    let shown = |i: usize| counters.get(i).copied().unwrap_or(false);
    let micros = |ms: Option<i64>| ms.map(|m| m.saturating_mul(1000));
    vec![
        text(rows.iter().map(|r| Some(r.group_id.as_str()))),
        text(rows.iter().map(|r| r.parent_id.as_deref())),
        text(rows.iter().map(|r| Some(r.state))),
        text(rows.iter().map(|r| Some(r.scheduling_policy))),
        Arc::new(Int32Array::from_iter(
            rows.iter().map(|r| Some(r.scheduling_weight)),
        )),
        Arc::new(Int32Array::from_iter(rows.iter().map(|r| Some(r.priority)))),
        text(rows.iter().map(|r| Some(r.workload_class))),
        Arc::new(Int32Array::from_iter(
            rows.iter().map(|r| Some(r.hard_concurrency_limit)),
        )),
        Arc::new(Int32Array::from_iter(
            rows.iter().map(|r| Some(r.max_queued)),
        )),
        long(rows.iter().map(|r| r.soft_memory_limit_bytes)),
        long(rows.iter().map(|r| r.per_query_memory_limit_bytes)),
        long(rows.iter().map(|r| r.scan_quota_bytes)),
        long(rows.iter().map(|r| r.scan_quota_period_s)),
        Arc::new(Int32Array::from_iter(
            rows.iter()
                .enumerate()
                .map(|(i, r)| shown(i).then_some(r.running)),
        )),
        Arc::new(Int32Array::from_iter(
            rows.iter()
                .enumerate()
                .map(|(i, r)| shown(i).then_some(r.queued)),
        )),
        long(
            rows.iter()
                .enumerate()
                .map(|(i, r)| shown(i).then_some(r.memory_reserved_bytes)),
        ),
        long(rows.iter().enumerate().map(|(i, r)| {
            if shown(i) {
                r.scan_bytes_in_window
            } else {
                None
            }
        })),
        Arc::new(
            TimestampMicrosecondArray::from_iter(
                rows.iter()
                    .enumerate()
                    .map(|(i, r)| micros(r.scan_window_resets_at_ms.filter(|_| shown(i)))),
            )
            .with_timezone("UTC"),
        ),
        text(
            rows.iter()
                .map(|r| (!r.config_etag.is_empty()).then_some(r.config_etag.as_str())),
        ),
        Arc::new(
            TimestampMicrosecondArray::from_iter(
                rows.iter()
                    .map(|r| micros((r.loaded_at_ms > 0).then_some(r.loaded_at_ms))),
            )
            .with_timezone("UTC"),
        ),
    ]
}

pub(crate) fn layout() -> Vec<(&'static str, Kind, bool)> {
    use Kind::{Int, Long, Text, Time};
    vec![
        ("group_id", Text, false),
        ("parent_id", Text, true),
        ("state", Text, false),
        ("scheduling_policy", Text, false),
        ("scheduling_weight", Int, false),
        ("priority", Int, false),
        ("workload_class", Text, false),
        ("hard_concurrency_limit", Int, false),
        ("max_queued", Int, false),
        ("soft_memory_limit_bytes", Long, true),
        ("per_query_memory_limit_bytes", Long, true),
        ("scan_quota_bytes", Long, true),
        ("scan_quota_period_s", Long, true),
        ("running", Int, true),
        ("queued", Int, true),
        ("memory_reserved_bytes", Long, true),
        ("scan_bytes_in_window", Long, true),
        ("scan_window_resets_at", Time, true),
        ("config_etag", Text, true),
        ("loaded_at", Time, true),
    ]
}
