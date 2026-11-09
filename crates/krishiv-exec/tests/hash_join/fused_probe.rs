// SPDX-License-Identifier: Apache-2.0
//! A filter fused into the probe (design P2-E3 U2,
//! `HashJoin::with_probe_filter`) joins exactly what the join of the
//! filtered input joins: every join type, NULL keys, `<=>` keys, the
//! residual, weighted probe rows, NULL predicate values, grace spill
//! (deferred probe rows) and the rows outer and anti joins emit at the end;
//! in the same order where the unfused order is deterministic.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_exec::testing::Pipe;
use krishiv_exec::{
    ExecConfig, FnPredicate, HashJoin, HashJoinNode, JoinSpec, JoinType, Predicate, Result,
};
use krishiv_types::batch::{Batch, Weights};
use proptest::prelude::*;

use super::util::*;
use super::{Case, LONG_A, Row, any_case, data, row, schema};

/// `v % 3 != 0`, NULL when `v == 4`.
fn keep(b: &RecordBatch) -> Result<BooleanArray> {
    let v = b.column(2).as_primitive::<Int64Type>();
    Ok(v.iter()
        .map(|v| v.and_then(|v| (v != 4).then_some(v % 3 != 0)))
        .collect())
}

fn predicate() -> Arc<dyn Predicate> {
    Arc::new(FnPredicate(keep))
}

/// Every output row as `a|b|...` with its weight (`*w`) unless 1, in
/// output order.
fn rows_in_order(batches: &[Batch]) -> Vec<String> {
    let mut rows = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            let mut row: Vec<String> = b.data().columns().iter().map(|c| cell(c, i)).collect();
            let w = match b.weights() {
                Weights::Const(w) => *w,
                Weights::Column(c) => c.value(i),
            };
            if w != 1 {
                row.push(format!("*{w}"));
            }
            rows.push(row.join("|"));
        }
    }
    rows
}

/// The probe input as batches of `batch_rows`, weighted 1, -1, 2, ...
/// when `weighted`.
fn probe_batches(probe: &[Row], batch_rows: usize, weighted: bool) -> Vec<Batch> {
    let plain = batches(&data(probe), batch_rows);
    if !weighted {
        return plain;
    }
    let mut k = 0i64;
    plain
        .into_iter()
        .map(|b| {
            let w: Vec<i64> = (0..b.num_rows())
                .map(|_| {
                    k += 1;
                    [1, -1, 2][usize::try_from(k % 3).unwrap()]
                })
                .collect();
            Batch::with_weights(b.into_parts().0, Weights::Column(Int64Array::from(w))).unwrap()
        })
        .collect()
}

/// The join of `case` with the filter as its own operator below the probe
/// (`fused = false`) or fused into the probe; the rows in output order and
/// the probe's `probe_spilled_rows`.
#[expect(clippy::too_many_arguments, reason = "one run of the comparison")]
async fn run_one(
    case: &Case,
    probe: &[Row],
    build: &[Row],
    batch_rows: usize,
    threads: usize,
    limit: usize,
    weighted: bool,
    fused: bool,
) -> (Vec<String>, u64) {
    let root = root(limit);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        batch_size: if probe.len() < 1000 { 5 } else { 1024 },
        join_row_layout: case.row_layout,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let mut join = HashJoin::try_new(case.spec(Some(scope.clone()))).unwrap();
    let input = scan(&schema(), probe_batches(probe, batch_rows, weighted));
    let probe_node = if fused {
        join = join.with_probe_filter(predicate()).unwrap();
        input
    } else {
        Arc::new(Pipe(
            input,
            krishiv_exec::Filter::new(schema(), predicate()),
        ))
    };
    let plan = Arc::new(HashJoinNode {
        join,
        build: scan(&schema(), batches(&data(build), batch_rows)),
        probe: probe_node,
    });
    let out = run(&runner_for(threads, limit), plan, &q).await.unwrap();
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0, "spill files are deleted");
    (
        rows_in_order(&out),
        metric(&q, "HashJoinProbe", "probe_spilled_rows"),
    )
}

/// The cases of [`any_case`] plus the existence join; a weighted probe
/// only where the join takes one (not the joins that return build rows).
fn fused_case() -> impl Strategy<Value = (Case, bool)> {
    (any_case(), any::<bool>(), any::<bool>()).prop_map(|(mut c, existence, weighted)| {
        if existence && c.join_type != JoinType::NullAwareAnti {
            c.join_type = JoinType::Existence;
        }
        let weighted = weighted && !c.join_type.builds_preserved();
        (c, weighted)
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]
    /// In memory: the same rows in the same order on one thread (but for
    /// the joins that return build rows), the same rows on several.
    #[test]
    fn a_fused_filter_joins_what_the_filtered_input_joins(
        (c, weighted) in fused_case(),
        probe in prop::collection::vec(row(6, 4), 0..80),
        build in prop::collection::vec(row(6, 4), 0..80),
        batch_rows in 1usize..17,
        threads in 1usize..4,
    ) {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let run = |fused| rt.block_on(run_one(
            &c, &probe, &build, batch_rows, threads, 64 * MIB, weighted, fused,
        ));
        let (mut expect, _) = run(false);
        let (mut got, _) = run(true);
        // The joins that return build rows emit them per probe driver as
        // the drivers finish: no order to compare.
        if threads > 1 || c.join_type.builds_preserved() {
            expect.sort();
            got.sort();
        }
        prop_assert_eq!(got, expect);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 4, max_shrink_iters: 16, ..ProptestConfig::default() })]
    /// Under a budget that forces grace spill (deferred probe rows,
    /// asserted): the same rows as the unfused join.
    #[test]
    fn a_fused_filter_joins_the_same_rows_through_grace_spill(
        (c, weighted) in fused_case(),
        probe in prop::collection::vec(row(20_000, 200), 10_000..15_000),
        build in prop::collection::vec(row(20_000, 200), 15_000..20_000),
        batch_rows in 64usize..512,
        threads in 1usize..4,
    ) {
        let mut c = c;
        if c.join_type == JoinType::NullAwareAnti {
            c.join_type = JoinType::LeftAnti;
        }
        c.keys = 2;
        c.null_safe.resize(2, false);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .start_paused(true)
            .build()
            .unwrap();
        let run = |fused| rt.block_on(run_one(
            &c, &probe, &build, batch_rows, threads, MIB, weighted, fused,
        ));
        let (mut expect, _) = run(false);
        let (mut got, deferred) = run(true);
        expect.sort();
        got.sort();
        prop_assert!(got == expect, "fused grace result differs");
        prop_assert!(deferred > 0, "no probe row was deferred to a spilled partition");
    }
}

/// A batch the filter empties is dropped, and a batch it keeps whole is
/// joined as it is: the same rows as the filtered input.
#[tokio::test]
async fn empty_and_whole_selections() {
    let c = super::case(JoinType::LeftOuter, 1, false, false);
    let mk = |k: i64, v: i64| Row {
        k1: Some(k),
        k2: Some(LONG_A),
        v,
    };
    // Batches of 2: [3, 6] all dropped, [1, 2] all kept, [4, 5] half.
    let probe = [mk(1, 3), mk(2, 6), mk(1, 1), mk(2, 2), mk(1, 4), mk(3, 5)];
    let build = [mk(1, 0), mk(2, 0)];
    for weighted in [false, true] {
        let (expect, _) = run_one(&c, &probe, &build, 2, 1, 64 * MIB, weighted, false).await;
        let (got, _) = run_one(&c, &probe, &build, 2, 1, 64 * MIB, weighted, true).await;
        assert_eq!(got, expect);
        assert_eq!(got.len(), 3, "{got:?}");
    }
}

/// A long string of row `i`: never inlined in its view.
fn long_string(i: usize) -> String {
    format!("a string too long to inline, row {i:08}")
}

/// The memory the output of a join over `Utf8View` probe strings holds, the
/// filter keeping one row in a hundred, fused or as its own operator.
async fn view_output_bytes(fused: bool) -> usize {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("s", DataType::Utf8View, false),
    ]));
    let rows = 10_000usize;
    let probe = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..rows).map(|i| i64::try_from(i).unwrap()),
            )),
            Arc::new(StringViewArray::from_iter_values(
                (0..rows).map(long_string),
            )),
        ],
    )
    .unwrap();
    let build = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..rows).map(|i| i64::try_from(i).unwrap()),
            )),
            Arc::new(StringViewArray::from_iter_values(
                (0..rows).map(|_| "b".to_owned()),
            )),
        ],
    )
    .unwrap();
    let every_hundredth: Arc<dyn Predicate> = Arc::new(FnPredicate(|b: &RecordBatch| {
        Ok(b.column(0)
            .as_primitive::<Int64Type>()
            .iter()
            .map(|k| k.map(|k| k % 100 == 0))
            .collect())
    }));
    let root = root(64 * MIB);
    let (_manager, scope) = spill();
    let q = query(&root, 1, ExecConfig::default());
    let mut join = HashJoin::try_new(JoinSpec {
        join_type: JoinType::Inner,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        single_match: false,
        spill: Some(scope),
    })
    .unwrap();
    let input = scan(&schema, vec![Batch::new(probe)]);
    let probe_node = if fused {
        join = join.with_probe_filter(every_hundredth).unwrap();
        input
    } else {
        Arc::new(Pipe(
            input,
            krishiv_exec::Filter::new(Arc::clone(&schema), every_hundredth),
        ))
    };
    let plan = Arc::new(HashJoinNode {
        join,
        build: scan(&schema, vec![Batch::new(build)]),
        probe: probe_node,
    });
    let out = run(&runner_for(1, 64 * MIB), plan, &q).await.unwrap();
    assert_eq!(out.iter().map(Batch::num_rows).sum::<usize>(), rows / 100);
    out.iter().map(|b| b.data().get_array_memory_size()).sum()
}

/// A selective fused filter compacts its probe strings as the `Filter` it
/// replaces does: the output does not hold the scan's whole string buffers,
/// which every breaker above would charge to the memory pool.
#[tokio::test]
async fn a_fused_filter_does_not_hold_the_scans_string_buffers() {
    let unfused = view_output_bytes(false).await;
    let fused = view_output_bytes(true).await;
    assert!(fused < unfused * 4, "fused {fused}, unfused {unfused}");
}
