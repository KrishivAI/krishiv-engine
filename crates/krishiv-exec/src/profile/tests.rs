// SPDX-License-Identifier: Apache-2.0

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

fn env() -> ExprEnv {
    ExprEnv {
        ansi: true,
        zone: Some(0),
        query_start_micros: 0,
        user: Arc::from("u"),
    }
}

/// `Limit(Union(OneRow, Limit(OneRow)))`, printed with `explain`.
fn plan(explain: ExplainNode) -> PhysicalPlan {
    let root = PhysNode::Limit {
        input: Box::new(PhysNode::Union {
            inputs: vec![
                PhysNode::OneRow,
                PhysNode::Limit {
                    input: Box::new(PhysNode::OneRow),
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
    let mut by_addr = crate::physical::node_ids(p.root())
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

#[test]
fn a_deep_plan_round_trips() {
    let mut root = PhysNode::OneRow;
    let mut explain = ExplainNode::leaf("OneRowRelation", "");
    for _ in 0..600 {
        root = PhysNode::Limit {
            input: Box::new(root),
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
            "unknown metric kind",
            Box::new(|v| {
                v["operators"][2]["extra"][0]["kind"] = "avg".into();
            }),
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
wall <t>, 1 pipeline, <n> drivers, <n> thread (most in one pipeline), 1 worker
Limit 5
:  · operator #1 Limit p0: in 1 rows (1 batch); out 1 rows (1 batch); compute <t> (max <t>, <n> driver); peak mem <mem>
+- Union
   :- OneRowRelation
   :     · source #2 OneRow p0: out 1 rows (1 batch); compute <t> (max <t>, <n> driver); peak mem <mem>; \
hash_entries=3 hash_buckets=4 (load 0.75) probe_lookups=4 probe_key_compares=6 (avg 1.50) max_probe_length=3 \
spill_bytes=<mem> hash_index=swiss
   +- Limit 1
      +- OneRowRelation
Result:
   sink   #0 ResultSink p0: in 1 rows (1 batch); compute <t> (max <t>, <n> driver); peak mem <mem>
Pipelines:
   p0: wall <t>, <n> driver, <n> thread: #2 OneRow -> #1 Limit -> #0 ResultSink
";
    assert_eq!(text, want);
    let plain = p.to_text(TextOptions::default());
    assert!(plain.contains("wall 3.0 ms"), "{plain}");
    assert!(plain.contains("compute 10.0 µs"), "{plain}");
    assert!(plain.contains("spill_bytes=2.0 KiB"), "{plain}");
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
