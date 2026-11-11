// SPDX-License-Identifier: Apache-2.0
//! Tests that runtime filters reach the scans of every root of a plan.

use std::collections::BTreeMap;

use arrow_schema::DataType;
use krishiv_io::{ColumnPath, FilterId};
use krishiv_plan::{ReadSource, Scan};

use super::*;
use crate::physical::ScanNode;

fn scan(id: u64) -> PhysNode {
    PhysNode::Scan(ScanNode {
        id: ScanId(id),
        scan: Scan {
            source: ReadSource::Table {
                name: vec!["t".to_owned()],
                options: BTreeMap::new(),
            },
            output: Vec::new(),
            version: None,
        },
        projection: Vec::new(),
        filter: None,
        runtime_filters: Vec::new(),
        row_stage: crate::operators::RowStageSettings::default(),
        row_id: false,
    })
}

fn filter_of(node: &PhysNode) -> Vec<FilterId> {
    match node {
        PhysNode::Scan(s) => s.runtime_filters.iter().map(|f| f.id).collect(),
        _ => Vec::new(),
    }
}

/// A filter reaches its scan in whichever tree holds it.
#[test]
fn filters_reach_the_scans_of_every_root() {
    let filter = |id| ScanRuntimeFilter {
        id: FilterId(id),
        column: ColumnPath::new("k"),
        key_type: DataType::Int64,
    };
    let mut filters = HashMap::from([(ScanId(0), vec![filter(10)]), (ScanId(1), vec![filter(11)])]);
    let (mut a, mut b) = (scan(0), scan(1));
    attach_runtime_filters(&mut filters, true, [&mut a, &mut b]);
    assert_eq!(filter_of(&a), [FilterId(10)]);
    assert_eq!(filter_of(&b), [FilterId(11)]);
    assert!(filters.is_empty());
}
