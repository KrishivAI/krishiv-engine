// SPDX-License-Identifier: Apache-2.0
//! Tests for the query profile: node numbering, shared plans and subqueries, the JSON round trip and corrupt documents.
#![expect(
    clippy::disallowed_methods,
    reason = "test-only fixture read; never runs on a compute worker"
)]

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use arrow_schema::Schema;

use super::text::human_bytes;
use super::*;
use crate::context::OperatorId;
use crate::metrics::{
    Metric, OperatorEntry, OperatorMetrics, OperatorRole, PipelineMetrics, QueryMetrics,
};
use crate::physical::{ExprEnv, PhysNode};
use krishiv_plan::{Expr, ExprNode};

fn env() -> ExprEnv {
    ExprEnv {
        ansi: true,
        legacy_statistical_aggregate: false,
        zone: Some(0),
        zone_name: None,
        query_start_micros: 0,
        user: Arc::from("u"),
        map_key_dedup_last_win: false,
        legacy_java_charsets: false,
        legacy_coding_error_action: false,
        legacy_time_parser: false,
        binary_output_style: None,
    }
}

/// `Limit(Union(OneRow, Limit(OneRow)))`, printed with `explain`.
fn plan(explain: ExplainNode) -> PhysicalPlan {
    let root = PhysNode::Limit {
        offset: 0,
        input: Box::new(PhysNode::Union {
            inputs: vec![
                PhysNode::OneRow,
                PhysNode::Limit {
                    input: Box::new(PhysNode::OneRow),
                    offset: 0,
                    limit: 1,
                },
            ],
        }),
        limit: 5,
    };
    PhysicalPlan::new(root, Arc::new(Schema::empty()), explain, env())
}

fn matching() -> ExplainNode {
    ExplainNode::new(
        "Limit",
        "5",
        vec![ExplainNode::new(
            "Union",
            "",
            vec![
                ExplainNode::leaf("OneRowRelation", ""),
                ExplainNode::new("Limit", "1", vec![ExplainNode::leaf("OneRowRelation", "")]),
            ],
        )],
    )
}

fn entry(name: &'static str, node: Option<u32>, role: OperatorRole, rows: u64) -> OperatorEntry {
    let mut metrics = OperatorMetrics {
        rows_in: rows,
        batches_in: 1,
        rows_out: rows,
        batches_out: 1,
        compute_time: Duration::from_micros(rows * 10),
        peak_reserved: rows * 100,
        ..OperatorMetrics::default()
    };
    metrics.as_driver();
    OperatorEntry {
        name: Cow::Borrowed(name),
        node,
        role,
        pipelines: vec![0],
        metrics,
        extra: Vec::new(),
    }
}

fn metrics() -> QueryMetrics {
    let mut m = QueryMetrics {
        wall_time: Duration::from_millis(3),
        workers: 1,
        ..QueryMetrics::default()
    };
    m.operators.insert(
        OperatorId(0),
        entry("ResultSink", None, OperatorRole::Sink, 1),
    );
    m.operators.insert(
        OperatorId(1),
        entry("Limit", Some(0), OperatorRole::Operator, 1),
    );
    let mut scan = entry("OneRow", Some(2), OperatorRole::Source, 1);
    scan.extra = vec![
        (Cow::Borrowed("hash_entries"), Metric::Count(3)),
        (Cow::Borrowed("hash_buckets"), Metric::Count(4)),
        (Cow::Borrowed("probe_lookups"), Metric::Count(4)),
        (Cow::Borrowed("probe_key_compares"), Metric::Count(6)),
        (Cow::Borrowed("max_probe_length"), Metric::Max(3)),
        (Cow::Borrowed("spill_bytes"), Metric::Count(2048)),
        (Cow::Borrowed("finalize_ns"), Metric::Count(2_500_000)),
        (
            Cow::Borrowed("hash_index"),
            Metric::Text("swiss".to_owned()),
        ),
    ];
    m.operators.insert(OperatorId(2), scan);
    m.pipelines.insert(
        0,
        PipelineMetrics {
            drivers: 1,
            threads: 1,
            started: Duration::from_millis(1),
            ended: Duration::from_millis(3),
            max_slice: Duration::from_micros(1_500),
            stalls: 2,
        },
    );
    m
}

#[test]
fn nodes_are_numbered_in_pre_order() {
    let tree = ProfileNode::of(&plan(matching()));
    let mut ids = Vec::new();
    tree.walk(&mut |n| ids.push((n.id, n.name.clone())));
    assert_eq!(
        ids,
        [
            (Some(0), "Limit".to_owned()),
            (Some(1), "Union".to_owned()),
            (Some(2), "OneRowRelation".to_owned()),
            (Some(3), "Limit".to_owned()),
            (Some(4), "OneRowRelation".to_owned()),
        ]
    );
    // The same numbering instantiation tags operators with.
    let p = plan(matching());
    let mut by_addr = crate::physical::node_ids([p.root()])
        .into_values()
        .collect::<Vec<_>>();
    by_addr.sort_unstable();
    assert_eq!(by_addr, [0, 1, 2, 3, 4]);
}

#[test]
fn subquery_headings_and_mismatches() {
    // A `Subquery` heading wraps an input without being a plan node.
    let wrapped = ExplainNode::new(
        "Limit",
        "5",
        vec![ExplainNode::new(
            "Subquery",
            "scalar",
            vec![ExplainNode::new(
                "Union",
                "",
                vec![
                    ExplainNode::leaf("OneRowRelation", ""),
                    ExplainNode::new("Limit", "1", vec![ExplainNode::leaf("OneRowRelation", "")]),
                ],
            )],
        )],
    );
    let tree = ProfileNode::of(&plan(wrapped));
    assert_eq!(tree.children[0].id, None);
    assert_eq!(tree.children[0].children[0].id, Some(1));
    // A node whose inputs disagree keeps its text, without ids below it.
    let short = ExplainNode::new(
        "Limit",
        "5",
        vec![ExplainNode::new(
            "Union",
            "",
            vec![ExplainNode::leaf("OneRowRelation", "")],
        )],
    );
    let tree = ProfileNode::of(&plan(short));
    assert_eq!(tree.children[0].id, Some(1));
    assert_eq!(tree.children[0].children[0].id, None);
}

#[test]
fn json_round_trips() {
    let p = QueryProfile::new(&plan(matching()), metrics(), true);
    let text = p.to_json();
    let back = QueryProfile::from_json(&text).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back, p);
    assert_eq!(back.to_json(), text);
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
    assert_eq!(v["format"], PROFILE_FORMAT);
    assert_eq!(v["version"], PROFILE_VERSION);
    assert_eq!(v["plan"][0]["children"], serde_json::json!([1]));
}

/// A shared plan's nodes are numbered before the root's, as instantiation
/// numbers them, and each tree keeps its own text.
#[test]
fn a_profile_numbers_the_shared_plans_first() {
    use crate::physical::{SharedId, SharedPlan};

    let read = PhysNode::SharedRead {
        id: SharedId(3),
        layout: Vec::new(),
    };
    let plan = PhysicalPlan::new(
        PhysNode::Limit {
            input: Box::new(read),
            offset: 0,
            limit: 1,
        },
        Arc::new(Schema::empty()),
        ExplainNode::new("Limit", "1", vec![ExplainNode::leaf("SharedRead", "#3")]),
        env(),
    )
    .with_shared(vec![SharedPlan {
        id: SharedId(3),
        plan: PhysNode::OneRow,
        explain: ExplainNode::leaf("OneRowRelation", ""),
        deterministic: true,
        ordered: false,
    }])
    .unwrap();
    let node = ProfileNode::of(&plan);
    assert_eq!((node.id, node.name.as_str()), (None, "WithShared"));
    let [body, root] = node.children.as_slice() else {
        panic!("{node:?}");
    };
    assert_eq!((body.id, body.detail.as_str()), (None, "#3"));
    assert_eq!(body.children[0].id, Some(0));
    assert_eq!(body.children[0].name, "OneRowRelation");
    assert_eq!(root.id, Some(1));
    assert_eq!(root.children[0].id, Some(2));
    let ids = crate::physical::node_ids(plan.roots());
    assert_eq!(ids[&std::ptr::from_ref(plan.root()).addr()], 1);
    assert_eq!(ids[&std::ptr::from_ref(&plan.shared()[0].plan).addr()], 0);
    let p = QueryProfile::new(&plan, QueryMetrics::default(), false);
    let back = QueryProfile::from_json(&p.to_json()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back, p);
}

/// The deepest plan a document holds: a body at the limit, below the two
/// headings of a shared plan.
#[test]
fn a_shared_plan_at_the_depth_limit_is_read() {
    let plan = PhysicalPlan::new(
        PhysNode::OneRow,
        Arc::new(Schema::empty()),
        ExplainNode::leaf("OneRowRelation", ""),
        env(),
    );
    let doc = QueryProfile::new(&plan, QueryMetrics::default(), false).to_json();
    let mut v: serde_json::Value = serde_json::from_str(&doc).unwrap();
    let n = 2 * krishiv_plan::MAX_DEPTH + 2;
    v["plan"] = (0..n)
        .map(|i| {
            let children: Vec<usize> = if i + 1 < n { vec![i + 1] } else { vec![] };
            serde_json::json!({"id": null, "name": "x", "detail": "", "children": children})
        })
        .collect();
    let read = QueryProfile::from_json(&v.to_string()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(read.plan.name, "x");
}

#[test]
fn a_deep_plan_round_trips() {
    let mut root = PhysNode::OneRow;
    let mut explain = ExplainNode::leaf("OneRowRelation", "");
    for _ in 0..600 {
        root = PhysNode::Limit {
            input: Box::new(root),
            offset: 0,
            limit: 1,
        };
        explain = ExplainNode::new("Limit", "1", vec![explain]);
    }
    let plan = PhysicalPlan::new(root, Arc::new(Schema::empty()), explain, env());
    let p = QueryProfile::new(&plan, QueryMetrics::default(), false);
    let back = QueryProfile::from_json(&p.to_json()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back, p);
}

#[test]
fn corrupt_documents_are_rejected() {
    let good = QueryProfile::new(&plan(matching()), metrics(), true).to_json_value();
    type Edit = Box<dyn Fn(&mut serde_json::Value)>;
    let cases: Vec<(&str, Edit)> = vec![
        ("format is not", Box::new(|v| v["format"] = "x".into())),
        (
            "version 2 is not supported",
            Box::new(|v| v["version"] = 2.into()),
        ),
        (
            "missing member \"query\"",
            Box::new(|v| {
                if let Some(o) = v.as_object_mut() {
                    o.remove("query");
                }
            }),
        ),
        (
            "wall_time_ns is not an unsigned integer",
            Box::new(|v| {
                v["query"]["wall_time_ns"] = (-1).into();
            }),
        ),
        (
            "unknown operator role",
            Box::new(|v| v["operators"][0]["role"] = "x".into()),
        ),
        (
            "a plan input is not a later node",
            Box::new(|v| {
                v["plan"][1]["children"] = serde_json::json!([0]);
            }),
        ),
        (
            "the input of two nodes",
            Box::new(|v| {
                v["plan"][3]["children"] = serde_json::json!([4]);
                v["plan"][1]["children"] = serde_json::json!([2, 4]);
            }),
        ),
        (
            "not one tree",
            Box::new(|v| v["plan"][1]["children"] = serde_json::json!([2])),
        ),
        (
            "the plan is too deep",
            Box::new(|v| {
                let n = 2 * krishiv_plan::MAX_DEPTH + 3;
                v["plan"] = (0..n)
                    .map(|i| {
                        let children: Vec<usize> = if i + 1 < n { vec![i + 1] } else { vec![] };
                        serde_json::json!({"id": null, "name": "x", "detail": "", "children": children})
                    })
                    .collect();
            }),
        ),
        (
            "operator 1 listed twice",
            Box::new(|v| {
                let first = v["operators"][1].clone();
                if let Some(a) = v["operators"].as_array_mut() {
                    a.push(first);
                }
            }),
        ),
        (
            "pipeline 0 listed twice",
            Box::new(|v| {
                let first = v["pipelines"][0].clone();
                if let Some(a) = v["pipelines"].as_array_mut() {
                    a.push(first);
                }
            }),
        ),
        (
            "id exceeds u32",
            Box::new(|v| v["operators"][0]["id"] = u64::MAX.into()),
        ),
        (
            "listed twice",
            Box::new(|v| {
                let first = v["operators"][2]["extra"][0].clone();
                if let Some(a) = v["operators"][2]["extra"].as_array_mut() {
                    a.push(first);
                }
            }),
        ),
    ];
    for (want, edit) in cases {
        let mut v = good.clone();
        edit(&mut v);
        let err = QueryProfile::from_json(&v.to_string())
            .err()
            .unwrap_or_else(|| panic!("accepted a document with {want}"));
        assert!(
            err.to_string().starts_with("invalid query profile: "),
            "{err}"
        );
        assert!(err.to_string().contains(want), "{want}: {err}");
    }
    let err = QueryProfile::from_json("{").err().map(|e| e.to_string());
    assert!(err.is_some_and(|e| e.contains("not JSON")));
}

#[test]
fn text_places_operators_under_their_nodes() {
    let p = QueryProfile::new(&plan(matching()), metrics(), true);
    let text = p.to_text(TextOptions::masked());
    let want = "\
== Physical Plan (analyzed) ==
wall <t>, 1 pipeline, <n> drivers, <n> threads (most in one pipeline), 1 worker
Limit 5
:  · operator #1 Limit p0: in 1 rows (1 batch); out 1 rows (1 batch); compute <t> (max <t>, <n> drivers); peak mem <mem>
+- Union
   :- OneRowRelation
   :     · source #2 OneRow p0: out 1 rows (1 batch); compute <t> (max <t>, <n> drivers); io <t> (max <t>); peak mem <mem>; \
hash_entries=3 hash_buckets=<n> probe_lookups=<n> probe_key_compares=<n> max_probe_length=<n> \
spill_bytes=<mem> finalize_ns=<t> hash_index=swiss
   +- Limit 1
      +- OneRowRelation
Result:
   sink   #0 ResultSink p0: in 1 rows (1 batch); compute <t> (max <t>, <n> drivers); peak mem <mem>
Pipelines:
   p0: wall <t>, <n> drivers, <n> threads: #2 OneRow -> #1 Limit -> #0 ResultSink
";
    assert_eq!(text, want);
    let plain = p.to_text(TextOptions::default());
    assert!(plain.contains("wall 3.0 ms"), "{plain}");
    assert!(plain.contains("compute 10.0 µs"), "{plain}");
    assert!(plain.contains("finalize_ns=2.5 ms"), "{plain}");
    assert!(plain.contains("spill_bytes=2.0 KiB"), "{plain}");
    assert!(plain.contains("hash_buckets=4 (load 0.75)"), "{plain}");
    assert!(
        plain.contains("probe_key_compares=6 (avg 1.50) max_probe_length=3"),
        "{plain}"
    );
    let undetailed = QueryProfile::new(&plan(matching()), metrics(), false);
    assert!(
        undetailed
            .to_text(TextOptions::masked())
            .contains("(probe lengths not collected)")
    );
}

#[test]
fn profiles_merge_across_workers() {
    let mut a = QueryProfile::new(&plan(matching()), metrics(), true);
    let b = QueryProfile::new(&plan(matching()), metrics(), false);
    a.merge(&b).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(a.metrics.workers, 2);
    assert!(!a.detailed);
    let scan = &a.metrics.operators[&OperatorId(2)];
    assert_eq!(scan.metrics.rows_out, 2);
    assert_eq!(scan.metrics.drivers, 2);
    assert_eq!(scan.extra("max_probe_length"), Some(&Metric::Max(3)));
    assert_eq!(scan.extra("hash_entries"), Some(&Metric::Count(6)));
    let other = QueryProfile::new(
        &PhysicalPlan::new(
            PhysNode::OneRow,
            Arc::new(Schema::empty()),
            ExplainNode::leaf("OneRowRelation", ""),
            env(),
        ),
        metrics(),
        true,
    );
    let err = a.merge(&other).err().map(|e| e.to_string());
    assert_eq!(
        err.as_deref(),
        Some("invalid query profile: cannot merge profiles of different plans")
    );
}

#[test]
fn bytes_are_human() {
    assert_eq!(human_bytes(0), "0 B");
    assert_eq!(human_bytes(1023), "1023 B");
    assert_eq!(human_bytes(1536), "1.5 KiB");
    assert_eq!(human_bytes(5 << 20), "5.0 MiB");
    assert_eq!(human_bytes(3 << 30), "3.0 GiB");
    assert_eq!(human_bytes(u64::MAX), "16777216.0 TiB");
}

/// The checked-in JSON Schema lists exactly the members the writer emits,
/// at every level (a drift test: a new member needs a schema change, and a
/// removed one a version bump).
/// The share is the probes' table time over the drivers' busy time, in the
/// JSON and the text; a query without a probe has none, and a document from
/// before the fields existed still reads.
#[test]
fn the_hash_probe_share_is_reported() {
    let mut m = metrics();
    assert_eq!(m.hash_probe_share(), None);
    m.drivers.busy = Duration::from_millis(400);
    assert_eq!(m.hash_probe_share(), None);
    let mut probe = entry("HashJoinProbe", Some(2), OperatorRole::Operator, 1);
    probe.extra = vec![(Cow::Borrowed("probe_table_ns"), Metric::Count(100_000_000))];
    m.operators.insert(OperatorId(9), probe);
    assert_eq!(m.hash_probe_share(), Some(0.25));

    let p = QueryProfile::new(&plan(matching()), m, true);
    let doc = p.to_json_value();
    assert_eq!(doc["query"]["busy_ns"], 400_000_000_u64);
    assert_eq!(doc["query"]["hash_probe_share"], 0.25);
    let text = p.to_text(TextOptions::default());
    assert!(
        text.contains("hash probe: 25.0% of 400.0 ms driver busy time"),
        "{text}"
    );
    let back = QueryProfile::from_json(&p.to_json()).map(|b| b.metrics.drivers.busy);
    assert_eq!(back.ok(), Some(Duration::from_millis(400)));

    let mut old = doc;
    let query = old["query"].as_object_mut();
    query.map(|q| q.remove("busy_ns"));
    let back = QueryProfile::from_json(&old.to_string()).map(|b| b.metrics.drivers.busy);
    assert_eq!(back.ok(), Some(Duration::ZERO));
}

/// Mean reads in flight is the scans' outstanding read time over the query's
/// wall time, in the JSON and the text; a query without a scan read has none.
#[test]
fn the_mean_reads_in_flight_is_reported() {
    let mut m = metrics();
    assert_eq!(m.scan_reads_in_flight_mean(), None);
    m.wall_time = Duration::from_millis(100);
    assert_eq!(m.scan_reads_in_flight_mean(), None);
    // Two scans: 150 ms and 100 ms of reads outstanding over 100 ms of wall time.
    for (id, ns) in [(7, 150_000_000_u64), (8, 100_000_000)] {
        let mut scan = entry("FileScan", None, OperatorRole::Source, 1);
        scan.extra = vec![(Cow::Borrowed("scan_read_inflight_ns"), Metric::Count(ns))];
        m.operators.insert(OperatorId(id), scan);
    }
    assert_eq!(m.scan_reads_in_flight_mean(), Some(2.5));
    // A zero wall time has no mean.
    let mut no_wall = m.clone();
    no_wall.wall_time = Duration::ZERO;
    assert_eq!(no_wall.scan_reads_in_flight_mean(), None);

    let p = QueryProfile::new(&plan(matching()), m, true);
    let doc = p.to_json_value();
    assert_eq!(doc["query"]["scan_reads_in_flight_mean"], 2.5);
    let text = p.to_text(TextOptions::default());
    assert!(
        text.contains("scan reads: 2.5 in flight on average"),
        "{text}"
    );
    // Masked text hides the run-dependent mean.
    let masked = p.to_text(TextOptions::masked());
    assert!(
        masked.contains("scan reads: <n> in flight on average"),
        "{masked}"
    );
    // Absent in a document written before the field existed: it still reads.
    let mut old = doc;
    old["query"]
        .as_object_mut()
        .unwrap()
        .remove("scan_reads_in_flight_mean");
    assert!(QueryProfile::from_json(&old.to_string()).is_ok());
}

/// A document of a later minor loads: unknown members are ignored and an
/// unknown metric kind is kept as text. A different major is refused.
#[test]
fn a_later_minor_is_read() {
    let p = QueryProfile::new(&plan(matching()), metrics(), true);
    let mut v = p.to_json_value();
    v["minor"] = (PROFILE_MINOR + 1).into();
    v["tasks"] = serde_json::json!([{"fragment": 3}]);
    v["query"]["new_counter"] = 5.into();
    v["operators"][2]["extra"][0]["kind"] = "avg".into();
    let back = QueryProfile::from_json(&v.to_string()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back.plan, p.plan);
    let (_, kept) = &back.metrics.operators[&OperatorId(2)].extra[0];
    assert_eq!(*kept, Metric::Text("3".to_owned()));
    v["version"] = 2.into();
    assert!(QueryProfile::from_json(&v.to_string()).is_err());
}

/// The spread of rows over the drivers is written and read back, and a
/// document without it (written before minor 1) reads as no spread.
#[test]
fn instances_round_trip() {
    let mut m = metrics();
    if let Some(e) = m.operators.get_mut(&OperatorId(1)) {
        e.metrics.drivers = 4;
        e.metrics.min_driver_rows_out = 0;
        e.metrics.max_driver_rows_out = 91_000;
    }
    let p = QueryProfile::new(&plan(matching()), m, false);
    let v = p.to_json_value();
    assert_eq!(
        v["operators"][1]["instances"],
        serde_json::json!({"count": 4, "min_rows_out": 0, "max_rows_out": 91000})
    );
    assert_eq!(QueryProfile::from_json(&v.to_string()).ok(), Some(p));
    let mut old = v;
    if let Some(o) = old["operators"][1].as_object_mut() {
        o.remove("instances");
    }
    let back = QueryProfile::from_json(&old.to_string()).unwrap_or_else(|e| panic!("{e}"));
    let e = &back.metrics.operators[&OperatorId(1)].metrics;
    assert_eq!((e.min_driver_rows_out, e.max_driver_rows_out), (0, 0));
}

/// `planning` round-trips and must be an object.
#[test]
fn planning_round_trips() {
    let mut p = QueryProfile::new(&plan(matching()), metrics(), false);
    p.planning = Some(serde_json::json!({"parsed": "p"}));
    let back = QueryProfile::from_json(&p.to_json()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back, p);
    let mut v = p.to_json_value();
    v["planning"] = "x".into();
    assert!(QueryProfile::from_json(&v.to_string()).is_err());
}

/// `trace` (minor 4) round-trips; a minor-3 document without it reads as `None`; a `trace` that
/// is not an object is refused.
#[test]
fn trace_round_trips_and_is_optional_and_checked() {
    let mut p = QueryProfile::new(&plan(matching()), metrics(), false);
    p.trace = Some(serde_json::json!({
        "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
        "span_id": "00f067aa0ba902b7"
    }));
    let back = QueryProfile::from_json(&p.to_json()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back, p);
    // A minor-3 document: no `trace` member.
    let mut v = p.to_json_value();
    v["minor"] = 3.into();
    if let Some(o) = v.as_object_mut() {
        o.remove("trace");
    }
    let old = QueryProfile::from_json(&v.to_string()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(old.trace, None);
    // `null` reads as absent; any other non-object is an error.
    v["trace"] = serde_json::Value::Null;
    assert_eq!(
        QueryProfile::from_json(&v.to_string())
            .map(|b| b.trace)
            .ok(),
        Some(None)
    );
    for bad in [
        serde_json::json!("x"),
        serde_json::json!([1]),
        serde_json::json!(7),
    ] {
        v["trace"] = bad;
        let err = QueryProfile::from_json(&v.to_string())
            .err()
            .map(|e| e.to_string());
        assert!(
            err.is_some_and(|m| m.contains("trace is not an object")),
            "a non-object trace is refused"
        );
    }
}

/// The usage block sums the operators' counters (T138).
#[test]
fn usage_equals_the_operator_sums() {
    let mut m = metrics();
    m.memory.peak_reserved = 4096;
    for (id, bytes, rows) in [(2, 700, 30), (1, 50, 4)] {
        let e = m.operators.get_mut(&OperatorId(id)).expect("operator");
        e.extra
            .push((Cow::Borrowed("scan_bytes"), Metric::Count(bytes)));
        e.extra
            .push((Cow::Borrowed("scan_rows_decoded"), Metric::Count(rows)));
        e.metrics.spilled_bytes = 10 * bytes;
    }
    let u = QueryUsage::of(&m);
    assert_eq!(u.fidelity, "measured");
    assert_eq!(u.wall, Duration::from_millis(3));
    assert_eq!(u.cpu, Duration::from_micros(30));
    assert_eq!(u.rows_out, 1);
    assert_eq!((u.scan_bytes, u.scan_rows), (750, 34));
    assert_eq!(u.spill_bytes, 7500);
    assert_eq!(u.peak_memory_bytes, 4096);
    let v = QueryProfile::new(&plan(matching()), m, false).to_json_value();
    assert_eq!(v["minor"], PROFILE_MINOR);
    assert_eq!(v["usage"]["scan"]["bytes"], 750);
    assert_eq!(v["usage"]["cpu_ns"], 30_000);
}

#[test]
fn schema_matches_the_writer() {
    use serde_json::Value;
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/reference/query-profile-v1.schema.json"
    );
    let schema: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_default())
        .unwrap_or_default();
    let mut profile = QueryProfile::new(&plan(with_facts()), metrics(), true);
    profile.hints = vec![Hint {
        code: HintCode::Spill,
        node: Some(0),
        operator: Some(OperatorId(1)),
        message: "m".to_owned(),
        evidence: vec![("n".to_owned(), 1)],
    }];
    profile.planning = Some(
        serde_json::json!({"parsed": "p", "analyzed": "a", "steps": [],
        "optimized": "o", "physical": "x"}),
    );
    profile.trace = Some(serde_json::json!({"trace_id": "t", "session_id": "s"}));
    profile.identity = Some(ProfileIdentity {
        principal: "user:a".into(),
        tags: [("team".to_owned(), "x".to_owned())].into(),
    });
    profile.scans = vec![ProfileScan {
        table: "c.s.t".to_owned(),
        rows: Some(10),
        equality: vec![ScanColumn {
            name: "d".to_owned(),
            distinct: Some(3),
        }],
    }];
    profile.planning_ns = Some(1_500);
    let doc = profile.to_json_value();
    let keys = |v: &Value| -> Vec<String> {
        let mut k: Vec<String> = v
            .as_object()
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        k.sort();
        k
    };
    let required = |s: &Value| -> Vec<String> {
        let mut k: Vec<String> = s["required"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        k.sort();
        k
    };
    // Every member the writer emits is a schema property; every required
    // property is emitted (members added within v1 are optional).
    let properties = |s: &Value| -> Vec<String> {
        let mut k: Vec<String> = s["properties"]
            .as_object()
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        k.sort();
        k
    };
    let check = |s: &Value, d: &Value| {
        assert_eq!(properties(s), keys(d), "properties vs writer");
        let k = keys(d);
        for r in required(s) {
            assert!(k.contains(&r), "{r} is required but not written");
        }
    };
    let props = &schema["properties"];
    check(&schema, &doc);
    assert_eq!(props["format"]["const"], PROFILE_FORMAT);
    assert_eq!(props["version"]["const"], PROFILE_VERSION);
    check(&props["query"], &doc["query"]);
    check(
        &props["query"]["properties"]["memory"],
        &doc["query"]["memory"],
    );
    check(&schema["$defs"]["planning"], &doc["planning"]);
    check(&props["trace"], &doc["trace"]);
    check(&schema["$defs"]["facts"], &doc["plan"][0]["facts"]);
    check(&schema["$defs"]["hint"], &doc["hints"][0]);
    check(&props["environment"], &doc["environment"]);
    check(&props["usage"], &doc["usage"]);
    check(
        &props["operators"]["items"]["properties"]["instances"],
        &doc["operators"][0]["instances"],
    );
    check(&props["usage"]["properties"]["scan"], &doc["usage"]["scan"]);
    check(&props["pipelines"]["items"], &doc["pipelines"][0]);
    check(&props["plan"]["items"], &doc["plan"][0]);
    let op = &props["operators"]["items"];
    check(op, &doc["operators"][0]);
    check(
        &op["properties"]["metrics"],
        &doc["operators"][0]["metrics"],
    );
    for variant in op["properties"]["extra"]["items"]["oneOf"]
        .as_array()
        .into_iter()
        .flatten()
    {
        assert_eq!(required(variant), keys(&doc["operators"][2]["extra"][0]));
    }
    // A document written before the members added within v1 still reads.
    let mut old = doc;
    if let Some(q) = old["query"].as_object_mut() {
        q.remove("memory");
    }
    for o in old["operators"].as_array_mut().into_iter().flatten() {
        if let Some(m) = o["metrics"].as_object_mut() {
            m.remove("spill_files");
            m.remove("spill_write_bytes");
        }
    }
    let back = QueryProfile::from_json(&old.to_string()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back.metrics.memory, crate::metrics::QueryMemory::default());
}

/// Merging documents with extreme counters saturates instead of
/// overflowing.
#[test]
fn merging_extreme_documents_saturates() {
    let mut m = metrics();
    if let Some(e) = m.operators.get_mut(&OperatorId(1)) {
        e.metrics.rows_out = u64::MAX;
        e.metrics.compute_time = Duration::MAX;
        e.extra = vec![(Cow::Borrowed("groups"), Metric::Count(u64::MAX))];
    }
    let doc = QueryProfile::new(&plan(matching()), m, true).to_json();
    let mut a = QueryProfile::from_json(&doc).unwrap_or_else(|e| panic!("{e}"));
    let b = a.clone();
    a.merge(&b).unwrap_or_else(|e| panic!("{e}"));
    let e = &a.metrics.operators[&OperatorId(1)];
    assert_eq!(e.metrics.rows_out, u64::MAX);
    assert_eq!(e.extra("groups"), Some(&Metric::Count(u64::MAX)));
    assert!(e.metrics.compute_time >= Duration::from_nanos(u64::MAX));
}

#[test]
fn profile_errors_carry_their_class() {
    let e = QueryProfile::from_json("[]")
        .err()
        .unwrap_or_else(|| ProfileError::new("x"));
    let k = krishiv_common::error::KrishivError::from(e);
    assert_eq!(
        k.error_class(),
        krishiv_common::error::ErrorClass::KrishivInvalidProfileDocument
    );
    assert!(k.to_string().contains("profile is not an object"), "{k}");
}

#[test]
fn a_minor_2_document_reads_pipeline_stalls_as_zero() {
    let p = QueryProfile::new(&plan(matching()), metrics(), true);
    let mut v = p.to_json_value();
    assert_eq!(v["pipelines"][0]["max_slice_ns"], 1_500_000);
    assert_eq!(v["pipelines"][0]["stalls"], 2);
    v["minor"] = 2.into();
    let pipe = v["pipelines"][0].as_object_mut().unwrap();
    pipe.remove("max_slice_ns");
    pipe.remove("stalls");
    let back = QueryProfile::from_json(&v.to_string()).unwrap();
    let read = &back.metrics.pipelines[&0];
    assert_eq!((read.max_slice, read.stalls), (Duration::ZERO, 0));
    assert_eq!(read.ended, Duration::from_millis(3));
}

#[test]
fn the_trace_member_round_trips_and_is_absent_without_one() {
    let mut p = QueryProfile::new(&plan(matching()), metrics(), true);
    assert!(p.to_json_value().get("trace").is_none());
    p.trace = Some(
        serde_json::json!({"trace_id": "sentinel-trace-7f3a", "session_id": "sentinel-session-9c1d"}),
    );
    let back = QueryProfile::from_json(&p.to_json()).expect("reads its own document");
    assert_eq!(back.trace, p.trace);
}

#[test]
fn identity_round_trips_and_a_shared_profile_drops_it_with_the_trace() {
    let mut p = QueryProfile::new(&plan(matching()), metrics(), true);
    p.trace = Some(
        serde_json::json!({"trace_id": "sentinel-trace-7f3a", "session_id": "sentinel-session-9c1d"}),
    );
    p.identity = Some(ProfileIdentity {
        principal: "sentinel-principal-ada".into(),
        tags: [("team".to_owned(), "sentinel-tag-data".to_owned())].into(),
    });
    let v = p.to_json_value();
    assert_eq!(v["usage"]["principal"], "sentinel-principal-ada");
    assert_eq!(v["usage"]["tags"]["team"], "sentinel-tag-data");
    assert_eq!(
        QueryProfile::from_json(&p.to_json())
            .expect("reads")
            .identity,
        p.identity
    );

    let shared = p.shareable();
    let text = shared.to_json();
    for marker in [
        "sentinel-principal-ada",
        "sentinel-tag-data",
        "sentinel-trace-7f3a",
        "sentinel-session-9c1d",
        "\"trace\"",
    ] {
        assert!(!text.contains(marker), "{marker} survived --share: {text}");
    }
    // Every number is kept.
    assert_eq!(shared.metrics, p.metrics);
    assert_eq!(shared.plan, p.plan);
}

#[test]
fn a_profile_without_usage_still_reads_with_no_identity() {
    let p = QueryProfile::new(&plan(matching()), metrics(), true);
    let mut v = p.to_json_value();
    v.as_object_mut().expect("object").remove("usage");
    let back = QueryProfile::from_json(&v.to_string()).expect("an older document reads");
    assert!(back.identity.is_none());
}

/// A resolved call of a canonical operator.
fn call(b: krishiv_plan::Builtin, args: Vec<Expr>) -> Expr {
    use krishiv_plan::{FunctionId, FunctionKind, ResolvedExpr, ResolvedFunction};
    Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Builtin(b),
            kind: FunctionKind::Scalar,
            args: args.into_iter().map(Arc::new).collect(),
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: krishiv_types::DataType::Boolean,
            nullable: true,
            deterministic: true,
        },
    )))
}

fn col(id: u64, name: &str) -> krishiv_plan::BoundColumn {
    krishiv_plan::BoundColumn {
        id: krishiv_plan::AttrId(id),
        name: name.to_owned(),
        data_type: krishiv_types::DataType::Long,
        nullable: true,
    }
}

fn col_expr(id: u64, name: &str) -> Expr {
    Expr::new(ExprNode::Resolved(krishiv_plan::ResolvedExpr::Column(col(
        id, name,
    ))))
}

fn lit(v: i64) -> Expr {
    Expr::literal(krishiv_plan::Literal::Long(v))
}

fn scan_of(table: &str, cols: &[(u64, &str)], filter: Option<Expr>) -> PhysNode {
    use crate::physical::{ScanFilter, ScanId, ScanNode};
    PhysNode::Scan(ScanNode {
        decided: None,
        id: ScanId(0),
        scan: krishiv_plan::Scan {
            source: krishiv_plan::ReadSource::Table {
                name: table.split('.').map(str::to_owned).collect(),
                options: std::collections::BTreeMap::new(),
            },
            output: cols.iter().map(|(i, n)| col(*i, n)).collect(),
            version: None,
        },
        projection: Vec::new(),
        filter: filter.map(|p| ScanFilter {
            layout: cols.iter().map(|(i, _)| krishiv_plan::AttrId(*i)).collect(),
            predicate: Arc::new(p),
            exact: false,
        }),
        runtime_filters: Vec::new(),
        row_stage: crate::operators::RowStageSettings::default(),
        row_id: false,
    })
}

fn plan_of(root: PhysNode) -> PhysicalPlan {
    PhysicalPlan::new(
        root,
        Arc::new(Schema::empty()),
        ExplainNode::leaf("Scan", ""),
        env(),
    )
}

#[test]
fn scans_name_the_tables_and_the_columns_compared_with_values() {
    use krishiv_plan::Builtin;
    // `d = 1 AND c IN (1, 2) AND e > 3 AND f = d` over `t`, the first two in a Filter node, `e = 5` pushed into the scan.
    let cols = [(1, "d"), (2, "c"), (3, "e"), (4, "f"), (5, "g")];
    let pushed = call(Builtin::Eq, vec![col_expr(5, "g"), lit(5)]);
    let filter = PhysNode::Filter {
        input: Box::new(scan_of("cat.db.t", &cols, Some(pushed))),
        layout: Vec::new(),
        predicate: Arc::new(call(
            Builtin::And,
            vec![
                call(Builtin::Eq, vec![lit(1), col_expr(1, "d")]),
                call(
                    Builtin::And,
                    vec![
                        call(Builtin::In, vec![col_expr(2, "c"), lit(1), lit(2)]),
                        call(
                            Builtin::And,
                            vec![
                                call(Builtin::Gt, vec![col_expr(3, "e"), lit(3)]),
                                call(Builtin::Eq, vec![col_expr(4, "f"), col_expr(1, "d")]),
                            ],
                        ),
                    ],
                ),
            ],
        )),
        subqueries: Vec::new(),
    };
    let stats = |_: &krishiv_plan::Scan| {
        Some(krishiv_optimizer::TableStats {
            rows: 1000,
            columns: [(
                "d".to_owned(),
                krishiv_optimizer::ColumnStats {
                    distinct: Some(365),
                    ..Default::default()
                },
            )]
            .into(),
        })
    };
    let scans = profile_scans(&plan_of(filter), &stats);
    assert_eq!(
        scans,
        vec![ProfileScan {
            table: "cat.db.t".to_owned(),
            rows: Some(1000),
            equality: vec![
                ScanColumn {
                    name: "c".to_owned(),
                    distinct: None
                },
                ScanColumn {
                    name: "d".to_owned(),
                    distinct: Some(365)
                },
                ScanColumn {
                    name: "g".to_owned(),
                    distinct: None
                },
            ],
        }]
    );
}

#[test]
fn scans_round_trip_and_an_older_document_has_none() {
    let mut profile = QueryProfile::new(&plan(matching()), metrics(), false);
    assert!(profile.to_json_value().get("scans").is_none());
    profile.scans = vec![ProfileScan {
        table: "a.b".to_owned(),
        rows: None,
        equality: vec![ScanColumn {
            name: "d".to_owned(),
            distinct: Some(7),
        }],
    }];
    let back = QueryProfile::from_json(&profile.to_json()).unwrap();
    assert_eq!(back.scans, profile.scans);
    // A document of minor 5 has no `scans`.
    let mut v = profile.to_json_value();
    v.as_object_mut().unwrap().remove("scans");
    v["minor"] = 5.into();
    assert!(
        QueryProfile::from_json(&v.to_string())
            .unwrap()
            .scans
            .is_empty()
    );
}

/// A mistyped or negative `distinct` or `rows` of a scan is a malformed document, not an unknown count; a missing
/// one is unknown.
#[test]
fn a_malformed_scan_count_is_rejected_and_a_missing_one_is_none() {
    let mut profile = QueryProfile::new(&plan(matching()), metrics(), false);
    profile.scans = vec![ProfileScan {
        table: "c.s.t".to_owned(),
        rows: Some(10),
        equality: vec![ScanColumn {
            name: "d".to_owned(),
            distinct: Some(3),
        }],
    }];
    let v = profile.to_json_value();
    for bad in [serde_json::Value::from("bad"), serde_json::Value::from(-1)] {
        let mut d = v.clone();
        d["scans"][0]["equality"][0]["distinct"] = bad.clone();
        assert!(
            QueryProfile::from_json(&d.to_string()).is_err(),
            "distinct {bad}"
        );
        let mut r = v.clone();
        r["scans"][0]["rows"] = bad.clone();
        assert!(
            QueryProfile::from_json(&r.to_string()).is_err(),
            "rows {bad}"
        );
    }
    let mut missing = v;
    missing["scans"][0].as_object_mut().unwrap().remove("rows");
    missing["scans"][0]["equality"][0]
        .as_object_mut()
        .unwrap()
        .remove("distinct");
    let back = QueryProfile::from_json(&missing.to_string()).unwrap();
    assert_eq!(
        back.scans,
        vec![ProfileScan {
            table: "c.s.t".to_owned(),
            rows: None,
            equality: vec![ScanColumn {
                name: "d".to_owned(),
                distinct: None,
            }],
        }]
    );
}

/// A table is named by its components, each backquoted when it is not a plain identifier, so `cat.db.a.b` split
/// two ways names two tables.
#[test]
fn a_scanned_table_keeps_the_boundaries_of_its_name() {
    let named = |parts: &[&str]| {
        let PhysNode::Scan(mut s) = scan_of("x", &[(1, "d")], None) else {
            unreachable!()
        };
        s.scan.source = krishiv_plan::ReadSource::Table {
            name: parts.iter().map(|p| (*p).to_owned()).collect(),
            options: std::collections::BTreeMap::new(),
        };
        profile_scans(&plan_of(PhysNode::Scan(s)), &|_| None)
            .into_iter()
            .map(|s| s.table)
            .collect::<Vec<_>>()
    };
    assert_eq!(named(&["cat", "db", "a.b"]), ["cat.db.`a.b`"]);
    assert_eq!(named(&["cat", "db.a", "b"]), ["cat.`db.a`.b"]);
    assert_eq!(named(&["c", "s", "a`b"]), ["c.s.`a``b`"]);
    assert_eq!(named(&["c", "s", "\u{c4}"]), ["c.s.`\u{c4}`"]);
    assert_eq!(
        named(&["spark_catalog", "default", "Ev_1"]),
        ["spark_catalog.default.Ev_1"]
    );
}

fn with_facts() -> ExplainNode {
    use crate::planner::{FilterGap, FilterGapReason, InputFacts, NodeFacts};
    let mut e = matching();
    e.facts = Some(Arc::new(NodeFacts {
        estimated_rows: Some(12.5),
        inputs: vec![InputFacts {
            estimated_rows: 1.0e6,
            statistics: false,
            tables_without_stats: vec!["t".to_owned()],
        }],
        filter_gaps: vec![FilterGap {
            key: 0,
            probe_key: "(a#1 + 0)".to_owned(),
            reason: FilterGapReason::NoScanColumn,
        }],
        tables_without_stats: vec!["t".to_owned()],
    }));
    e
}

/// Facts and hints round trip; equality and `merge` ignore facts, so a worker's profile from the plan codec
/// (no facts) merges with the coordinator's; a minor-5 document (neither member) still loads.
#[test]
fn facts_and_hints_round_trip_and_do_not_affect_equality() {
    let mut p = QueryProfile::new(&plan(with_facts()), metrics(), true);
    assert!(p.plan.facts.is_some());
    p.hints = vec![Hint {
        code: HintCode::Skew,
        node: Some(0),
        operator: Some(OperatorId(2)),
        message: "m".to_owned(),
        evidence: vec![("drivers".to_owned(), 4)],
    }];
    let text = p.to_json();
    let back = QueryProfile::from_json(&text).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back.plan.facts, p.plan.facts);
    assert_eq!(back.hints, p.hints);
    assert_eq!(back.to_json(), text);
    // Equality ignores facts.
    let bare = QueryProfile::new(&plan(matching()), metrics(), true);
    assert_eq!(bare.plan, p.plan);
    let mut merged = QueryProfile::new(&plan(with_facts()), metrics(), true);
    merged.merge(&bare).unwrap_or_else(|e| panic!("{e}"));
    // A minor-5 document has neither member.
    let mut v = p.to_json_value();
    v["minor"] = 5.into();
    v.as_object_mut().expect("object").remove("hints");
    v["plan"][0]
        .as_object_mut()
        .expect("object")
        .remove("facts");
    let old = QueryProfile::from_json(&v.to_string()).unwrap_or_else(|e| panic!("{e}"));
    assert!(old.hints.is_empty());
    assert!(old.plan.facts.is_none());
}

#[test]
fn text_lists_hints_only_when_there_are_some() {
    let mut p = QueryProfile::new(&plan(matching()), metrics(), true);
    assert!(!p.to_text(TextOptions::masked()).contains("== Hints =="));
    p.hints = vec![Hint {
        code: HintCode::Spill,
        node: Some(0),
        operator: None,
        message: "Limit spilled".to_owned(),
        evidence: Vec::new(),
    }];
    let text = p.to_text(TextOptions::masked());
    assert!(
        text.contains("== Hints ==\n[SPILL] node 0 Limit: Limit spilled\n"),
        "{text}"
    );
}

/// Join inputs are positional (probe, build): one malformed element drops them all instead of shifting the rest,
/// which would attribute the build input's facts to the probe.
#[test]
fn a_malformed_join_input_drops_all_inputs_rather_than_shifting_them() {
    let good = serde_json::json!({"estimated_rows": 10, "statistics": false, "tables_without_stats": ["t"]});
    let bad = serde_json::json!({"estimated_rows": "x", "statistics": true});
    let both = serde_json::json!({"inputs": [good, good]});
    let facts = super::facts_json::facts_of(&both).expect("facts");
    assert_eq!(facts.inputs.len(), 2);
    assert_eq!(facts.inputs[1].tables_without_stats, vec!["t".to_owned()]);
    let mixed = serde_json::json!({"inputs": [bad, good]});
    let facts = super::facts_json::facts_of(&mixed).expect("facts");
    assert!(facts.inputs.is_empty());
}

#[test]
fn planning_ns_round_trips_and_is_absent_when_unknown() {
    let mut profile = QueryProfile::new(&plan(with_facts()), metrics(), false);
    let doc = profile.to_json_value();
    assert!(doc["usage"].get("planning_ns").is_none());
    assert_eq!(
        QueryProfile::from_json(&profile.to_json())
            .unwrap()
            .planning_ns,
        None
    );
    profile.planning_ns = Some(35_100_000);
    let doc = profile.to_json_value();
    assert_eq!(doc["usage"]["planning_ns"], 35_100_000);
    assert_eq!(
        QueryProfile::from_json(&profile.to_json())
            .unwrap()
            .planning_ns,
        Some(35_100_000)
    );
    // A mistyped member is refused, as for the other usage members a reader knows.
    let mut bad = doc;
    bad["usage"]["planning_ns"] = "x".into();
    assert!(QueryProfile::from_json(&bad.to_string()).is_err());
}
