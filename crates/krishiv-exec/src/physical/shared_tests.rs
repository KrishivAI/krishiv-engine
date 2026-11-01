// SPDX-License-Identifier: Apache-2.0
//! Shared plans as plan data: the checks of `with_shared` and the walks over
//! a plan's trees.

use std::sync::Arc;

use arrow_schema::{DataType, Schema};
use krishiv_common::error::ErrorClass;
use krishiv_io::{ColumnPath, FilterId};
use krishiv_plan::{ReadSource, Scan};

use super::*;
use crate::planner::{ExplainNode, MemoryTables};
use crate::values::{LocalValues, ValueService as _};

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
    }
}

fn read(id: u64) -> PhysNode {
    PhysNode::SharedRead {
        id: SharedId(id),
        layout: Vec::new(),
    }
}

fn shared(id: u64, plan: PhysNode) -> SharedPlan {
    SharedPlan {
        id: SharedId(id),
        plan,
        explain: ExplainNode::leaf("Body", format!("{id}")),
        deterministic: true,
        ordered: false,
    }
}

fn plan(root: PhysNode) -> PhysicalPlan {
    PhysicalPlan::new(
        root,
        Arc::new(Schema::empty()),
        ExplainNode::leaf("Root", ""),
        env(),
    )
}

#[test]
fn a_plan_reads_only_shared_plans_listed_before_it() {
    let good = plan(read(2))
        .with_shared(vec![shared(1, PhysNode::OneRow), shared(2, read(1))])
        .unwrap();
    assert_eq!(good.shared().len(), 2);
    let cases = [
        ("no such plan", plan(read(1)).with_shared(Vec::new())),
        (
            "a later plan",
            plan(PhysNode::OneRow)
                .with_shared(vec![shared(1, read(2)), shared(2, PhysNode::OneRow)]),
        ),
        (
            "itself",
            plan(PhysNode::OneRow).with_shared(vec![shared(1, read(1))]),
        ),
        (
            "one id twice",
            plan(PhysNode::OneRow).with_shared(vec![
                shared(1, PhysNode::OneRow),
                shared(1, PhysNode::OneRow),
            ]),
        ),
        (
            "below the root",
            plan(PhysNode::Limit {
                input: Box::new(read(9)),
                offset: 0,
                limit: 1,
            })
            .with_shared(vec![shared(1, PhysNode::OneRow)]),
        ),
    ];
    for (name, got) in cases {
        let e = got.err().unwrap_or_else(|| panic!("{name} was accepted"));
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{name}");
    }
}

#[test]
fn roots_list_the_shared_plans_then_the_root() {
    let p = plan(read(2))
        .with_shared(vec![shared(1, PhysNode::OneRow), shared(2, read(1))])
        .unwrap();
    let roots: Vec<&PhysNode> = p.roots().collect();
    assert_eq!(roots, [&PhysNode::OneRow, &read(1), &read(2)]);
    let none = plan(PhysNode::OneRow);
    assert_eq!(none.roots().count(), 1);
}

#[test]
fn explain_prints_the_shared_plans_first() {
    let none = plan(PhysNode::OneRow);
    assert_eq!(none.explain_all(), *none.explain());
    assert_eq!(none.to_string(), none.explain().to_string());
    let p = plan(read(1))
        .with_shared(vec![shared(1, PhysNode::OneRow)])
        .unwrap();
    let all = p.explain_all();
    assert_eq!(all.name, WITH_SHARED);
    let names: Vec<(&str, &str)> = all
        .children
        .iter()
        .map(|c| (c.name.as_str(), c.detail.as_str()))
        .collect();
    assert_eq!(names, [(SHARED_PLAN, "#1"), ("Root", "")]);
    assert_eq!(all.children[0].children[0].name, "Body");
    assert_eq!(p.to_string(), all.to_string());
    // The root's own tree stays the root's.
    assert_eq!(p.explain().name, "Root");
}

#[test]
fn a_write_keeps_the_shared_plans() {
    let p = plan(read(1))
        .with_shared(vec![shared(1, PhysNode::OneRow)])
        .unwrap()
        .into_write(WritePlan {
            format: "parquet".to_owned(),
            options: std::collections::BTreeMap::new(),
            store: "file://".to_owned(),
            dir: "d".to_owned(),
            job_uuid: "j".to_owned(),
            max_records_per_file: 1,
            target_file_bytes: 1,
            partition_by: Vec::new(),
        })
        .unwrap();
    assert_eq!(p.shared().len(), 1);
}

/// A scan declaring runtime filter `id` to the worker's values.
fn filtered_scan(id: u64) -> PhysNode {
    PhysNode::Scan(ScanNode {
        id: ScanId(0),
        scan: Scan {
            source: ReadSource::Table {
                name: vec!["t".to_owned()],
                options: std::collections::BTreeMap::new(),
            },
            output: Vec::new(),
            version: None,
        },
        projection: Vec::new(),
        filter: None,
        runtime_filters: vec![ScanRuntimeFilter {
            id: FilterId(id),
            column: ColumnPath::new("k"),
            key_type: DataType::Int64,
        }],
        row_stage: crate::operators::RowStageSettings::default(),
    })
}

/// Until the executor runs shared plans, instantiating a read fails with an
/// internal error; the filters of a shared plan's scans are declared first.
#[test]
fn instantiation_declares_a_shared_plans_filters() {
    let values = Arc::new(LocalValues::new());
    let worker =
        WorkerContext::new(Arc::new(MemoryTables::new())).with_values(Arc::clone(&values) as _);
    let p = plan(read(1))
        .with_shared(vec![shared(1, filtered_scan(7))])
        .unwrap();
    // The filter of a scan inside the shared plan is declared before anything is built (the instantiation then fails on the missing table).
    let e = instantiate(&p, &worker).err().unwrap();
    // The empty table provider has no `t`: the shared scan fails before the plan's schema is compared.
    assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    assert_eq!(values.filter(FilterId(7)).unwrap().id(), FilterId(7));
}
