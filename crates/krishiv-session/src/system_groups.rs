// SPDX-License-Identifier: Apache-2.0
//! `system.resource_groups` (P3-T066): one row per resource group with its limits and live counters.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, TimestampMicrosecondArray};

use crate::system::{Kind, Sources, long, text};

/// The rows, for a caller with `Monitor` on `system` only.
pub(crate) fn columns(sources: &Sources) -> Vec<ArrayRef> {
    // Group names often name tenants: only a caller with `Monitor` on `system` lists them.
    // (The design's own-groups view for other callers is not built yet.)
    let rows = match (&sources.groups, sources.monitor.sees_all()) {
        (Some(groups), true) => groups.rows(),
        _ => Vec::new(),
    };
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
        Arc::new(Int32Array::from_iter(rows.iter().map(|r| Some(r.running)))),
        Arc::new(Int32Array::from_iter(rows.iter().map(|r| Some(r.queued)))),
        long(rows.iter().map(|r| Some(r.memory_reserved_bytes))),
        long(rows.iter().map(|r| r.scan_bytes_in_window)),
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
        ("running", Int, false),
        ("queued", Int, false),
        ("memory_reserved_bytes", Long, false),
        ("scan_bytes_in_window", Long, true),
        ("scan_window_resets_at", Time, true),
        ("config_etag", Text, false),
        ("loaded_at", Time, false),
    ]
}
