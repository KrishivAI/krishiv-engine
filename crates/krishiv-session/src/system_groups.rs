// SPDX-License-Identifier: Apache-2.0
//! `system.resource_groups` (P3-T066): one row per resource group with its limits and live counters.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, TimestampMicrosecondArray};

use crate::system::{Kind, Sources, long, text};

/// The rows a caller sees. With `Monitor` on `system`: every group. Without it (design §5.4):
/// the groups holding one of the caller's running queries, and their ancestors, whose subtree
/// counters are totals over other tenants' work and show as null; the caller's own group keeps
/// them. (A queued query takes its group when it starts, so it shows from then on.)
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
        .filter(|q| q.state == "running")
        .filter_map(|q| q.resource_group)
        .collect();
    rows.into_iter()
        .filter_map(|r| {
            let holds = own.contains(&r.group_id);
            let ancestor = own.iter().any(|g| {
                g.strip_prefix(r.group_id.as_str())
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
                rows.iter().map(|r| micros(r.scan_window_resets_at_ms)),
            )
            .with_timezone("UTC"),
        ),
        text(rows.iter().map(|r| Some(r.config_etag.as_str()))),
        Arc::new(
            TimestampMicrosecondArray::from_iter(rows.iter().map(|r| micros(Some(r.loaded_at_ms))))
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
        ("config_etag", Text, false),
        ("loaded_at", Time, false),
    ]
}
