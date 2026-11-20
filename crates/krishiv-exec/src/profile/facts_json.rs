// SPDX-License-Identifier: Apache-2.0
//! The profile document's minor-7 members: a plan node's `facts` and the query's `hints`. A reader that
//! does not know them ignores them; this reader skips a malformed one rather than refuse the document.

use serde_json::{Value, json};

use super::hints::{Hint, HintCode};
use crate::context::OperatorId;
use crate::planner::{FilterGap, FilterGapReason, InputFacts, NodeFacts};

pub(super) fn facts_json(f: &NodeFacts) -> Value {
    json!({
        "estimated_rows": f.estimated_rows,
        "inputs": f.inputs.iter().map(|i| json!({
            "estimated_rows": i.estimated_rows,
            "statistics": i.statistics,
            "tables_without_stats": i.tables_without_stats,
        })).collect::<Vec<_>>(),
        "filter_gaps": f.filter_gaps.iter().map(|g| json!({
            "key": g.key,
            "probe_key": g.probe_key,
            "reason": match g.reason {
                FilterGapReason::Disabled => "disabled",
                FilterGapReason::NoScanColumn => "no_scan_column",
                FilterGapReason::Blocked => "blocked",
            },
        })).collect::<Vec<_>>(),
        "tables_without_stats": f.tables_without_stats,
    })
}

fn strings(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

pub(super) fn facts_of(v: &Value) -> Option<NodeFacts> {
    let o = v.as_object()?;
    // Inputs are positional (probe, build): one malformed element drops them all rather than shifting the rest.
    let inputs = o
        .get("inputs")
        .and_then(Value::as_array)
        .and_then(|a| {
            a.iter()
                .map(|i| {
                    Some(InputFacts {
                        estimated_rows: i.get("estimated_rows")?.as_f64()?,
                        statistics: i.get("statistics")?.as_bool()?,
                        tables_without_stats: strings(i.get("tables_without_stats")),
                    })
                })
                .collect::<Option<Vec<_>>>()
        })
        .unwrap_or_default();
    let filter_gaps = o
        .get("filter_gaps")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|g| {
                    Some(FilterGap {
                        key: usize::try_from(g.get("key")?.as_u64()?).ok()?,
                        probe_key: g.get("probe_key")?.as_str()?.to_owned(),
                        reason: match g.get("reason")?.as_str()? {
                            "disabled" => FilterGapReason::Disabled,
                            "no_scan_column" => FilterGapReason::NoScanColumn,
                            _ => FilterGapReason::Blocked,
                        },
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Some(NodeFacts {
        estimated_rows: o.get("estimated_rows").and_then(Value::as_f64),
        inputs,
        filter_gaps,
        tables_without_stats: strings(o.get("tables_without_stats")),
    })
}

pub(super) fn hints_json(hints: &[Hint]) -> Value {
    Value::Array(
        hints
            .iter()
            .map(|h| {
                json!({
                    "code": h.code.as_str(),
                    "node": h.node,
                    "operator": h.operator.map(|o| o.0),
                    "message": h.message,
                    "evidence": h.evidence.iter().map(|(k, v)| (k.clone(), Value::from(*v)))
                        .collect::<serde_json::Map<_, _>>(),
                })
            })
            .collect(),
    )
}

pub(super) fn hints_of(v: &Value) -> Vec<Hint> {
    let Some(a) = v.as_array() else {
        return Vec::new();
    };
    a.iter()
        .filter_map(|h| {
            Some(Hint {
                code: HintCode::parse(h.get("code")?.as_str()?)?,
                node: h
                    .get("node")
                    .and_then(Value::as_u64)
                    .and_then(|n| u32::try_from(n).ok()),
                operator: h
                    .get("operator")
                    .and_then(Value::as_u64)
                    .and_then(|n| u32::try_from(n).ok())
                    .map(OperatorId),
                message: h.get("message")?.as_str()?.to_owned(),
                evidence: h
                    .get("evidence")
                    .and_then(Value::as_object)
                    .map(|e| {
                        e.iter()
                            .filter_map(|(k, v)| Some((k.clone(), v.as_u64()?)))
                            .collect()
                    })
                    .unwrap_or_default(),
            })
        })
        .collect()
}
