// SPDX-License-Identifier: Apache-2.0
//! The output filter (`aggregate/output_filter.rs`): a `HAVING` over the
//! values evaluated before the keys are decoded keeps exactly the groups
//! the filter keeps, in memory and after a spill, in both table layouts.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{BooleanArray, RecordBatch};
use arrow_schema::{Field, Schema};

use super::super::output_filter::OutputFilter;
use super::*;
use crate::operators::FnPredicate;

/// `sum(x) > 340_000` over the sum column alone (NULL sums fail).
fn having() -> OutputFilter {
    let schema = Arc::new(Schema::new(vec![Field::new("sum", DataType::Int64, true)]));
    let predicate = FnPredicate(|b: &RecordBatch| {
        Ok(b.column(0)
            .as_primitive::<Int64Type>()
            .iter()
            .map(|v| v.map(|v| v > 340_000))
            .collect::<BooleanArray>())
    });
    OutputFilter::new(Arc::new(predicate), vec![2], schema)
}

fn filtered(input: Arc<dyn ExecNode>, sp: AggregateSpec) -> Arc<dyn ExecNode> {
    let agg = HashAggregate::with_output_filter(HashAggregate::new(sp).unwrap(), having());
    AggregateNode::new(input, agg)
}

/// The rows of `all` (`k|count|sum`) whose sum passes.
fn passing(all: &[String]) -> Vec<String> {
    all.iter()
        .filter(|r| {
            r.split('|')
                .nth(2)
                .and_then(|s| s.parse::<i64>().ok())
                .is_some_and(|s| s > 340_000)
        })
        .cloned()
        .collect()
}

#[tokio::test]
async fn an_output_filter_keeps_exactly_the_passing_groups() {
    let (s, b) = grouped_input(20_000, 500);
    let aggs = || vec![AggregateExpr::count_star("c"), agg(&Sum, 1, &s)];
    let mut all = rows(
        &exec(
            node(scan(&s, b.clone()), spec(&s, vec![0], aggs())),
            Opts::default(),
        )
        .await
        .unwrap()
        .0,
    );
    all.sort();
    let want = passing(&all);
    assert!(!want.is_empty() && want.len() < all.len(), "{}", want.len());
    let mut got = rows(
        &exec(
            filtered(scan(&s, b.clone()), spec(&s, vec![0], aggs())),
            Opts::default(),
        )
        .await
        .unwrap()
        .0,
    );
    got.sort();
    assert_eq!(got, want);
    // Spilled: the disk phase's output passes through the filter too.
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], aggs());
    sp.spill = Some(scope);
    let o = Opts {
        limit: 256 << 10,
        ..Opts::default()
    };
    let (out, m) = exec(filtered(scan(&s, b), sp), o).await.unwrap();
    assert!(extra(&m, "HashAggregate", "spill_runs") > 0);
    let mut got = rows(&out);
    got.sort();
    assert_eq!(got, want);
}
