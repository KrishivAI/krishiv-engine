// SPDX-License-Identifier: Apache-2.0
//! The columns a `ROLLUP`, `CUBE` or `GROUPING SETS` query reads: the expansion is attributed
//! column by column (a grouping-set projection reads what its expressions reference; the null
//! fill, the grouping id and the set number read nothing), so a column the query never names is
//! not asked.

use super::super::dsl::{avg, sum};
use super::*;
use crate::Grouping;

/// `SELECT items FROM lineitem GROUP BY grouping`.
fn grouped(grouping: Grouping, items: Vec<Expr>) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::new(LogicalPlan::table(["lineitem"])),
        grouping,
        aggregates: items.into_iter().map(Arc::new).collect(),
    })
}

fn keys(names: &[&str]) -> Vec<Arc<Expr>> {
    names.iter().map(|n| Arc::new(c(n))).collect()
}

/// The lineitem columns the analysed `plan` reads.
fn lineitem_reads(plan: LogicalPlan) -> Vec<String> {
    let a = run(plan, &ctx()).unwrap();
    let s = a
        .securables
        .iter()
        .find(|s| s.securable.to_string() == "table `spark_catalog`.`default`.`lineitem`")
        .unwrap();
    s.columns.iter().cloned().collect()
}

#[test]
fn rollup_reads_its_keys_and_aggregate_inputs() {
    let plan = grouped(
        Grouping::Rollup(keys(&["l_returnflag", "l_linestatus"])),
        vec![
            c("l_returnflag"),
            c("l_linestatus"),
            sum(c("l_quantity")),
            f("grouping", vec![c("l_returnflag")]),
        ],
    );
    assert_eq!(
        lineitem_reads(plan),
        ["l_linestatus", "l_quantity", "l_returnflag"]
    );
}

#[test]
fn cube_reads_its_keys_and_aggregate_inputs() {
    let plan = grouped(
        Grouping::Cube(keys(&["l_returnflag", "l_shipmode"])),
        vec![
            c("l_shipmode"),
            avg(c("l_discount")),
            f("grouping_id", vec![]),
        ],
    );
    assert_eq!(
        lineitem_reads(plan),
        ["l_discount", "l_returnflag", "l_shipmode"]
    );
}

#[test]
fn grouping_sets_read_every_key_of_every_set() {
    // `l_tax` is in the second set only and is never selected: it is still read.
    let plan = grouped(
        Grouping::GroupingSets {
            exprs: keys(&["l_returnflag", "l_tax"]),
            sets: vec![
                keys(&["l_returnflag"]),
                keys(&["l_returnflag", "l_tax"]),
                vec![],
            ],
        },
        vec![
            c("l_returnflag"),
            count_star(),
            f("grouping", vec![c("l_tax")]),
            f("grouping_id", vec![c("l_returnflag"), c("l_tax")]),
        ],
    );
    assert_eq!(lineitem_reads(plan), ["l_returnflag", "l_tax"]);
}

#[test]
fn a_computed_grouping_key_reads_its_inputs() {
    let plan = grouped(
        Grouping::Rollup(vec![Arc::new(add(c("l_tax"), c("l_discount")))]),
        vec![count_star(), f("grouping_id", vec![])],
    );
    assert_eq!(lineitem_reads(plan), ["l_discount", "l_tax"]);
}

#[test]
fn repeated_grouping_sets_read_only_their_keys() {
    // Equal sets number their rows (`_gen_grouping_pos`): the number reads nothing.
    let plan = grouped(
        Grouping::GroupingSets {
            exprs: keys(&["l_shipmode"]),
            sets: vec![keys(&["l_shipmode"]), keys(&["l_shipmode"])],
        },
        vec![c("l_shipmode"), sum(c("l_quantity"))],
    );
    assert_eq!(lineitem_reads(plan), ["l_quantity", "l_shipmode"]);
}
