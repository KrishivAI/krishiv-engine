// SPDX-License-Identifier: Apache-2.0
//! The profile's JSON document (format `krishiv.query-profile`, version 1).
//!
//! Times are integer nanoseconds (`*_ns`), sizes integer bytes. Members
//! appear in a fixed order; operators and pipelines are listed by id. A
//! reader rejects another format or version, a missing or mistyped member
//! and an unknown enum value, so a document either round-trips exactly or
//! does not load.

use std::borrow::Cow;
use std::time::Duration;

use serde_json::{Map, Value, json};

use super::{PROFILE_FORMAT, PROFILE_VERSION, ProfileError, ProfileNode, QueryProfile};
use crate::context::OperatorId;
use crate::metrics::{
    DriverStats, Metric, OperatorEntry, OperatorMetrics, OperatorRole, PipelineMetrics,
    QueryMemory, QueryMetrics,
};

type R<T> = Result<T, ProfileError>;

fn ns(d: Duration) -> Value {
    Value::from(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

impl QueryProfile {
    /// The profile as a JSON value.
    #[must_use]
    pub fn to_json_value(&self) -> Value {
        let m = &self.metrics;
        let d = &m.drivers;
        json!({
            "format": PROFILE_FORMAT,
            "version": PROFILE_VERSION,
            "detailed": self.detailed,
            "query": {
                "wall_time_ns": ns(m.wall_time),
                "workers": m.workers,
                "drivers": d.drivers,
                "batches": d.batches,
                "yields": d.yields,
                "max_slice_ns": ns(d.max_slice),
                "stalls": d.stalls,
                "busy_ns": ns(d.busy),
                "hash_probe_share": m.hash_probe_share(),
                "memory": {
                    "limit_bytes": m.memory.limit,
                    "peak_reserved_bytes": m.memory.peak_reserved,
                },
            },
            "pipelines": m.pipelines.iter().map(|(id, p)| json!({
                "id": id,
                "drivers": p.drivers,
                "threads": p.threads,
                "started_ns": ns(p.started),
                "ended_ns": ns(p.ended),
            })).collect::<Vec<_>>(),
            "plan": plan_json(&self.plan),
            "operators": m.operators.iter().map(|(id, e)| operator_json(*id, e)).collect::<Vec<_>>(),
        })
    }

    /// The profile as one line of JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        self.to_json_value().to_string()
    }

    /// Reads a profile [`to_json`](Self::to_json) wrote.
    ///
    /// # Errors
    /// [`ProfileError`] for malformed JSON, another format or version, or a
    /// missing, mistyped or out-of-range member.
    pub fn from_json(text: &str) -> R<Self> {
        let v: Value =
            serde_json::from_str(text).map_err(|e| ProfileError::new(format!("not JSON: {e}")))?;
        let o = obj(&v, "profile")?;
        if str_of(o, "format")? != PROFILE_FORMAT {
            return Err(ProfileError::new(format!("format is not {PROFILE_FORMAT}")));
        }
        let version = u64_of(o, "version")?;
        if version != PROFILE_VERSION {
            return Err(ProfileError::new(format!(
                "version {version} is not supported (this build reads {PROFILE_VERSION})"
            )));
        }
        let detailed = member(o, "detailed")?
            .as_bool()
            .ok_or_else(|| ProfileError::new("detailed is not a boolean"))?;
        let q = obj(member(o, "query")?, "query")?;
        let mut metrics = QueryMetrics {
            wall_time: dur_of(q, "wall_time_ns")?,
            workers: u64_of(q, "workers")?,
            drivers: DriverStats {
                drivers: u64_of(q, "drivers")?,
                batches: u64_of(q, "batches")?,
                yields: u64_of(q, "yields")?,
                max_slice: dur_of(q, "max_slice_ns")?,
                stalls: u64_of(q, "stalls")?,
                // Absent in a profile written before the field existed.
                busy: if q.contains_key("busy_ns") {
                    dur_of(q, "busy_ns")?
                } else {
                    Duration::ZERO
                },
            },
            memory: memory_of(q)?,
            ..QueryMetrics::default()
        };
        for p in arr(o, "pipelines")? {
            let p = obj(p, "pipeline")?;
            let id = u32_of(p, "id")?;
            let pm = PipelineMetrics {
                drivers: u64_of(p, "drivers")?,
                threads: u64_of(p, "threads")?,
                started: dur_of(p, "started_ns")?,
                ended: dur_of(p, "ended_ns")?,
            };
            if metrics.pipelines.insert(id, pm).is_some() {
                return Err(ProfileError::new(format!("pipeline {id} listed twice")));
            }
        }
        let plan = plan_of(member(o, "plan")?)?;
        for e in arr(o, "operators")? {
            let (id, entry) = operator_of(e)?;
            if metrics.operators.insert(id, entry).is_some() {
                return Err(ProfileError::new(format!("operator {} listed twice", id.0)));
            }
        }
        Ok(Self {
            plan,
            metrics,
            detailed,
        })
    }
}

/// The plan as a flat list in pre-order, each node naming its inputs by
/// position: a deep plan stays a shallow document.
fn plan_json(root: &ProfileNode) -> Value {
    let mut out = Vec::new();
    let mut stack = vec![(root, None::<usize>)];
    let mut children: Vec<Vec<usize>> = Vec::new();
    while let Some((n, parent)) = stack.pop() {
        let at = out.len();
        if let Some(p) = parent {
            children[p].push(at);
        }
        out.push(n);
        children.push(Vec::new());
        stack.extend(n.children.iter().rev().map(|c| (c, Some(at))));
    }
    Value::Array(
        out.iter()
            .zip(children)
            .map(|(n, c)| {
                json!({
                    "id": n.id,
                    "name": n.name,
                    "detail": n.detail,
                    "children": c,
                })
            })
            .collect(),
    )
}

fn operator_json(id: OperatorId, e: &OperatorEntry) -> Value {
    let m = &e.metrics;
    json!({
        "id": id.0,
        "name": e.name,
        "node": e.node,
        "role": e.role.as_str(),
        "pipelines": e.pipelines,
        "metrics": {
            "drivers": m.drivers,
            "rows_in": m.rows_in,
            "batches_in": m.batches_in,
            "rows_out": m.rows_out,
            "batches_out": m.batches_out,
            "compute_time_ns": ns(m.compute_time),
            "max_driver_compute_time_ns": ns(m.max_driver_compute_time),
            "blocked_on_io_ns": ns(m.blocked_on_io),
            "max_driver_blocked_on_io_ns": ns(m.max_driver_blocked_on_io),
            "blocked_on_memory_ns": ns(m.blocked_on_memory),
            "blocked_on_sink_ns": ns(m.blocked_on_sink),
            "morsels": m.morsels,
            "peak_reserved_bytes": m.peak_reserved,
            "spills": m.spills,
            "spilled_bytes": m.spilled_bytes,
            "spill_files": m.spill_files,
            "spill_write_bytes": m.spill_write_bytes,
        },
        "extra": e.extra.iter().map(|(name, v)| match v {
            Metric::Count(n) => json!({"name": name, "kind": "count", "value": n}),
            Metric::Max(n) => json!({"name": name, "kind": "max", "value": n}),
            Metric::Text(t) => json!({"name": name, "kind": "text", "value": t}),
        }).collect::<Vec<_>>(),
    })
}

/// The deepest plan a document may hold: the IR's limit, doubled for the
/// headings `EXPLAIN` adds (a subquery's), and the two above a shared plan.
const MAX_PLAN_DEPTH: usize = 2 * krishiv_plan::MAX_DEPTH + 2;

/// Reads [`plan_json`]'s list: node 0 is the root, every other node is
/// the input of exactly one node listed before it.
fn plan_of(v: &Value) -> R<ProfileNode> {
    let list = v
        .as_array()
        .ok_or_else(|| ProfileError::new("plan is not an array"))?;
    let mut nodes = Vec::with_capacity(list.len());
    // Each node's depth (inputs come after their parent, so a parent's
    // depth is known first): a flat list could describe a chain deeper
    // than any plan, which the recursive tree must not get.
    let mut depth = vec![1_usize; list.len()];
    let mut inputs = Vec::with_capacity(list.len());
    let mut used = vec![false; list.len()];
    for (at, n) in list.iter().enumerate() {
        let o = obj(n, "plan node")?;
        let id = match member(o, "id")? {
            Value::Null => None,
            _ => Some(u32_of(o, "id")?),
        };
        let mut mine = Vec::new();
        for c in arr(o, "children")? {
            let c = c
                .as_u64()
                .and_then(|c| usize::try_from(c).ok())
                .filter(|&c| c > at && c < list.len())
                .ok_or_else(|| ProfileError::new("a plan input is not a later node"))?;
            if std::mem::replace(&mut used[c], true) {
                return Err(ProfileError::new("a plan node is the input of two nodes"));
            }
            depth[c] = depth[at].saturating_add(1);
            if depth[c] > MAX_PLAN_DEPTH {
                return Err(ProfileError::new("the plan is too deep"));
            }
            mine.push(c);
        }
        nodes.push(Some(ProfileNode {
            id,
            name: str_of(o, "name")?.to_owned(),
            detail: str_of(o, "detail")?.to_owned(),
            children: Vec::new(),
        }));
        inputs.push(mine);
    }
    if nodes.is_empty() || used.iter().skip(1).any(|u| !u) {
        return Err(ProfileError::new("the plan is not one tree"));
    }
    // Inputs come later in the list: assemble from the back.
    for at in (0..nodes.len()).rev() {
        let children = inputs[at]
            .iter()
            .map(|&c| {
                nodes[c]
                    .take()
                    .ok_or_else(|| ProfileError::new("a plan input is used twice"))
            })
            .collect::<R<Vec<_>>>()?;
        if let Some(n) = nodes[at].as_mut() {
            n.children = children;
        }
    }
    nodes[0]
        .take()
        .ok_or_else(|| ProfileError::new("the plan has no root"))
}

fn operator_of(v: &Value) -> R<(OperatorId, OperatorEntry)> {
    let o = obj(v, "operator")?;
    let id = OperatorId(u32_of(o, "id")?);
    let node = match member(o, "node")? {
        Value::Null => None,
        _ => Some(u32_of(o, "node")?),
    };
    let role = match str_of(o, "role")? {
        "source" => OperatorRole::Source,
        "operator" => OperatorRole::Operator,
        "sink" => OperatorRole::Sink,
        r => return Err(ProfileError::new(format!("unknown operator role {r:?}"))),
    };
    let pipelines = arr(o, "pipelines")?
        .iter()
        .map(|p| {
            p.as_u64()
                .and_then(|p| u32::try_from(p).ok())
                .ok_or_else(|| ProfileError::new("a pipeline id is not a u32"))
        })
        .collect::<R<Vec<_>>>()?;
    let m = obj(member(o, "metrics")?, "metrics")?;
    let metrics = OperatorMetrics {
        drivers: u64_of(m, "drivers")?,
        rows_in: u64_of(m, "rows_in")?,
        batches_in: u64_of(m, "batches_in")?,
        rows_out: u64_of(m, "rows_out")?,
        batches_out: u64_of(m, "batches_out")?,
        compute_time: dur_of(m, "compute_time_ns")?,
        max_driver_compute_time: dur_of(m, "max_driver_compute_time_ns")?,
        blocked_on_io: dur_of(m, "blocked_on_io_ns")?,
        max_driver_blocked_on_io: dur_of(m, "max_driver_blocked_on_io_ns")?,
        blocked_on_memory: dur_of(m, "blocked_on_memory_ns")?,
        blocked_on_sink: dur_of(m, "blocked_on_sink_ns")?,
        morsels: u64_of(m, "morsels")?,
        peak_reserved: u64_of(m, "peak_reserved_bytes")?,
        spills: u64_of(m, "spills")?,
        spilled_bytes: u64_of(m, "spilled_bytes")?,
        // Added within v1: absent in documents written before them.
        spill_files: opt_u64(m, "spill_files")?.unwrap_or(0),
        spill_write_bytes: opt_u64(m, "spill_write_bytes")?.unwrap_or(0),
    };
    let mut extra = Vec::new();
    for x in arr(o, "extra")? {
        let x = obj(x, "extra metric")?;
        let name = str_of(x, "name")?.to_owned();
        if extra.iter().any(|(n, _)| *n == name) {
            return Err(ProfileError::new(format!(
                "extra metric {name:?} listed twice"
            )));
        }
        let value = match str_of(x, "kind")? {
            "count" => Metric::Count(u64_of(x, "value")?),
            "max" => Metric::Max(u64_of(x, "value")?),
            "text" => Metric::Text(str_of(x, "value")?.to_owned()),
            k => return Err(ProfileError::new(format!("unknown metric kind {k:?}"))),
        };
        extra.push((Cow::Owned(name), value));
    }
    Ok((
        id,
        OperatorEntry {
            name: Cow::Owned(str_of(o, "name")?.to_owned()),
            node,
            role,
            pipelines,
            metrics,
            extra,
        },
    ))
}

/// The query's memory (added within v1: absent in earlier documents).
fn memory_of(q: &Map<String, Value>) -> R<QueryMemory> {
    let Some(v) = q.get("memory") else {
        return Ok(QueryMemory::default());
    };
    let m = obj(v, "memory")?;
    Ok(QueryMemory {
        limit: opt_u64(m, "limit_bytes")?,
        peak_reserved: u64_of(m, "peak_reserved_bytes")?,
    })
}

/// An optional unsigned member: absent or `null` is `None`.
fn opt_u64(o: &Map<String, Value>, key: &str) -> R<Option<u64>> {
    match o.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| ProfileError::new(format!("{key} is not an unsigned integer"))),
    }
}

fn obj<'a>(v: &'a Value, what: &str) -> R<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| ProfileError::new(format!("{what} is not an object")))
}

fn member<'a>(o: &'a Map<String, Value>, key: &str) -> R<&'a Value> {
    o.get(key)
        .ok_or_else(|| ProfileError::new(format!("missing member {key:?}")))
}

fn str_of<'a>(o: &'a Map<String, Value>, key: &str) -> R<&'a str> {
    member(o, key)?
        .as_str()
        .ok_or_else(|| ProfileError::new(format!("{key} is not a string")))
}

fn u64_of(o: &Map<String, Value>, key: &str) -> R<u64> {
    member(o, key)?
        .as_u64()
        .ok_or_else(|| ProfileError::new(format!("{key} is not an unsigned integer")))
}

fn u32_of(o: &Map<String, Value>, key: &str) -> R<u32> {
    u32::try_from(u64_of(o, key)?).map_err(|_| ProfileError::new(format!("{key} exceeds u32")))
}

fn dur_of(o: &Map<String, Value>, key: &str) -> R<Duration> {
    Ok(Duration::from_nanos(u64_of(o, key)?))
}

fn arr<'a>(o: &'a Map<String, Value>, key: &str) -> R<&'a Vec<Value>> {
    member(o, key)?
        .as_array()
        .ok_or_else(|| ProfileError::new(format!("{key} is not an array")))
}
