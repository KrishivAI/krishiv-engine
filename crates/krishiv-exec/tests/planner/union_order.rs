// SPDX-License-Identifier: Apache-2.0
//! `UNION ALL` emits its branches' rows in branch order: a fan-in runs
//! its pipelines one after another, in pipeline order, when none of
//! their shared operators needs a cross-driver flush, so the union
//! reads as Spark's partition-ordered collect does. Repeated: branches
//! running together interleave batches by arrival.

use super::{cell, env, run};

/// `SELECT 1 UNION ALL SELECT 2` reads `1` then `2`, and `LIMIT 1` over
/// it reads `1`: the branches in order, every run.
#[test]
fn union_all_emits_its_branches_in_order() {
    let env = env();
    let plan = env.plan("SELECT 1 AS x UNION ALL SELECT 2 AS x").unwrap();
    let limited = env
        .plan("SELECT 1 AS x UNION ALL SELECT 2 AS x LIMIT 1")
        .unwrap();
    for _ in 0..20 {
        let mut rows = Vec::new();
        for b in run(&plan).unwrap() {
            for r in 0..b.num_rows() {
                rows.push(cell(b.column(0).as_ref(), r));
            }
        }
        assert_eq!(rows, ["1", "2"]);
        let mut first = Vec::new();
        for b in run(&limited).unwrap() {
            for r in 0..b.num_rows() {
                first.push(cell(b.column(0).as_ref(), r));
            }
        }
        assert_eq!(first, ["1"]);
    }
}
