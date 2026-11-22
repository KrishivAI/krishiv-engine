// SPDX-License-Identifier: Apache-2.0
//! `extras` on hand-built inputs: filter shapes the batches before
//! `ShareCte` never leave (adjacent filters merge), so no SQL query reaches
//! them.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_plan::{AttrId, BoundColumn, Builtin, Expr, LogicalPlan, PlanNode, ReadSource, Scan};
use krishiv_types::DataType;

use super::extras;
use crate::expr::{call, column};

fn int(id: u64, name: &str) -> BoundColumn {
    BoundColumn {
        id: AttrId(id),
        name: name.to_owned(),
        data_type: DataType::Integer,
        nullable: true,
    }
}

fn scan(output: Vec<BoundColumn>) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Scan(Scan {
        source: ReadSource::Table {
            name: vec!["t".to_owned()],
            options: BTreeMap::new(),
        },
        output,
        version: None,
    })))
}

fn filter(input: &Arc<LogicalPlan>, predicate: Arc<Expr>) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(PlanNode::Filter {
        input: Arc::clone(input),
        predicate,
    }))
}

fn not_null(c: &BoundColumn) -> Arc<Expr> {
    call(Builtin::IsNotNull, vec![column(c)], DataType::Boolean)
}

/// The weak side's filter sits below a filter of the strong side's own:
/// every conjunct of the upper filter is an extra, matched one level down.
#[test]
fn a_weak_filter_below_the_strong_ones_own_is_matched_one_level_down() {
    let (a, b) = (int(1, "a"), int(2, "b"));
    let base = scan(vec![a.clone(), b.clone()]);
    let weak = filter(&base, not_null(&a));
    let strong = filter(&filter(&base, not_null(&a)), not_null(&b));
    assert_eq!(extras(&weak, &strong), Some(vec![not_null(&b)]));
}

/// The weak side filters by a conjunct the strong side lacks: no extras.
#[test]
fn a_weak_conjunct_the_strong_side_lacks_refuses() {
    let (a, b) = (int(1, "a"), int(2, "b"));
    let base = scan(vec![a.clone(), b.clone()]);
    let weak = filter(&base, not_null(&a));
    let strong = filter(&base, not_null(&b));
    assert_eq!(extras(&weak, &strong), None);
}
