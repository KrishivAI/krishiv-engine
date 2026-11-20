// SPDX-License-Identifier: Apache-2.0
//! The as-of merge (Spark 4.3's `SortMergeAsOfJoinExec`) against a nested-loop oracle: both
//! directions, exact and strict, inner and left, with and without a residual, NULL values at batch
//! edges, keys spanning several left batches, groups that spill; a group without a residual is
//! read once; Spark's forward distance is evaluated on the matched and the next passing row only.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_exec::{
    AsOfMergeDirection, AsOfMergeSpec, ExecConfig, JoinSpec, JoinType, Predicate, Result,
    SortMergeJoin, SortMergeJoinNode,
};
use krishiv_memory::TreePool;
use proptest::prelude::*;

use super::util::*;

/// One row: key `k`, as-of value `t`, payload `w`.
#[derive(Clone, Debug)]
struct Row {
    k: Option<i64>,
    t: Option<i64>,
    w: i64,
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("t", DataType::Int64, true),
        Field::new("w", DataType::Int64, false),
    ]))
}

fn data(rows: &[Row]) -> RecordBatch {
    let col = |f: &dyn Fn(&Row) -> Option<i64>| {
        Arc::new(Int64Array::from(rows.iter().map(f).collect::<Vec<_>>())) as _
    };
    RecordBatch::try_new(
        schema(),
        vec![col(&|r| r.k), col(&|r| r.t), col(&|r| Some(r.w))],
    )
    .unwrap()
}

/// `(left.w + right.w) % 3 != 0`, NULL when the sum is a multiple of 5.
#[derive(Debug)]
struct Residual;

impl Residual {
    fn eval(l: i64, r: i64) -> bool {
        (l + r) % 5 != 0 && (l + r) % 3 != 0
    }
}

impl Predicate for Residual {
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray> {
        let l = batch.column(2).as_primitive::<Int64Type>();
        let r = batch.column(5).as_primitive::<Int64Type>();
        Ok(l.iter()
            .zip(r.iter())
            .map(|(l, r)| {
                let s = l.unwrap_or(0) + r.unwrap_or(0);
                (s % 5 != 0).then_some(s % 3 != 0)
            })
            .collect())
    }
}

/// A distance check that raises on a pair whose right payload is `POISON`.
#[derive(Debug)]
struct Poison;

const POISON: i64 = 999;

impl Predicate for Poison {
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray> {
        let r = batch.column(5).as_primitive::<Int64Type>();
        if r.iter().any(|w| w == Some(POISON)) {
            return Err(krishiv_common::error::InternalError {
                message: "poisoned distance".to_owned(),
            }
            .into());
        }
        Ok(BooleanArray::from(vec![true; batch.num_rows()]))
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    join_type: JoinType,
    backward: bool,
    exact: bool,
    residual: bool,
}

impl Case {
    fn passes(&self, l: i64, r: i64) -> bool {
        match (self.backward, self.exact) {
            (true, true) => r <= l,
            (true, false) => r < l,
            (false, true) => r >= l,
            (false, false) => r > l,
        }
    }

    /// Each left row with its nearest right row, rendered like [`render`].
    fn oracle(&self, left: &[Row], right: &[Row]) -> Vec<String> {
        let fmt = |r: &Row| {
            let s = |v: Option<i64>| v.map_or("∅".to_owned(), |v| v.to_string());
            format!("{}|{}|{}", s(r.k), s(r.t), r.w)
        };
        let mut out = Vec::new();
        for l in left {
            let best = l.t.and_then(|lt| {
                let candidates = right.iter().filter(|r| {
                    l.k.is_some()
                        && r.k == l.k
                        && r.t.is_some_and(|rt| self.passes(lt, rt))
                        && (!self.residual || Residual::eval(l.w, r.w))
                });
                if self.backward {
                    candidates.max_by_key(|r| r.t)
                } else {
                    candidates.min_by_key(|r| r.t)
                }
            });
            match best {
                Some(r) => out.push(format!("{}|{}", fmt(l), fmt(r))),
                None if self.join_type == JoinType::LeftOuter => {
                    out.push(format!("{}|∅|∅|∅", fmt(l)));
                }
                None => {}
            }
        }
        out.sort();
        out
    }

    fn join(
        &self,
        distance: Option<Arc<dyn Predicate>>,
        scope: krishiv_memory::SpillScope,
    ) -> SortMergeJoin {
        let spec = JoinSpec {
            join_type: self.join_type,
            probe_schema: schema(),
            build_schema: schema(),
            probe_keys: vec![0],
            build_keys: vec![0],
            null_safe: vec![false],
            filter: self
                .residual
                .then(|| Arc::new(Residual) as Arc<dyn Predicate>),
            single_match: false,
            spill: Some(scope),
        };
        let as_of = AsOfMergeSpec {
            left_column: 1,
            right_column: 1,
            direction: if self.backward {
                AsOfMergeDirection::Backward
            } else {
                AsOfMergeDirection::Forward
            },
            allow_exact: self.exact,
            distance: None,
        };
        SortMergeJoin::try_new_with_distance(spec, Some(as_of), distance).unwrap()
    }
}

/// What a run returned and its merge's counters.
struct Run {
    rows: Vec<String>,
    right_rows: u64,
    groups_spilled: u64,
}

/// Runs `case` with `batch_rows`-row inputs and `group_bytes` before a group spills.
async fn run_case(
    case: Case,
    left: &[Row],
    right: &[Row],
    batch_rows: usize,
    group_bytes: usize,
    distance: Option<Arc<dyn Predicate>>,
) -> Result<Run> {
    let limit = 64 << 20;
    let root: Arc<TreePool> = root(limit);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        batch_size: 5,
        join_merge_buffer_bytes: group_bytes,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let plan: Arc<dyn krishiv_exec::ExecNode> = Arc::new(SortMergeJoinNode {
        join: Arc::new(case.join(distance, scope.clone())),
        probe: scan(&schema(), batches(&data(left), batch_rows)),
        build: scan(&schema(), batches(&data(right), batch_rows)),
    });
    let out = run(&runner_for(1, limit), plan, &q).await;
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0, "spill files are deleted");
    let out = out?;
    Ok(Run {
        rows: render(&out),
        right_rows: metric(&q, "SortMergeAsOfJoin", "as_of_right_rows"),
        groups_spilled: metric(&q, "SortMergeAsOfJoin", "groups_spilled"),
    })
}

fn row(k: Option<i64>, t: Option<i64>, w: i64) -> Row {
    Row { k, t, w }
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for join_type in [JoinType::Inner, JoinType::LeftOuter] {
        for backward in [true, false] {
            for exact in [true, false] {
                for residual in [false, true] {
                    out.push(Case {
                        join_type,
                        backward,
                        exact,
                        residual,
                    });
                }
            }
        }
    }
    out
}

/// NULL keys and values, NULLs at batch edges, equal values, an empty group after its NULLs, a
/// key whose left rows span several batches.
fn edge_rows() -> (Vec<Row>, Vec<Row>) {
    let mut left = vec![
        row(Some(1), None, 1),
        row(Some(1), None, 2),
        row(Some(1), Some(5), 3),
        row(None, Some(5), 4),
        row(Some(2), Some(1), 5),
        row(Some(3), Some(7), 6),
    ];
    // Key 4 spans several left batches (batches of 3 rows).
    left.extend((0..20).map(|i| row(Some(4), Some(i * 3), 10 + i)));
    let mut right = vec![
        row(Some(1), None, 100),
        row(Some(1), Some(5), 101),
        row(Some(1), Some(4), 102),
        row(Some(1), Some(6), 103),
        row(None, Some(5), 104),
        // Key 2 has only a NULL value: an empty group once it is skipped.
        row(Some(2), None, 105),
        row(Some(3), Some(1), 106),
    ];
    right.extend((0..25).map(|i| row(Some(4), Some(i * 2 + 1), 200 + i)));
    (left, right)
}

#[tokio::test]
async fn as_of_edge_cases_match_the_oracle() {
    let (left, right) = edge_rows();
    for case in cases() {
        for batch_rows in [1, 2, 3, 7, 100] {
            for group_bytes in [usize::MAX, 64] {
                let got = run_case(case, &left, &right, batch_rows, group_bytes, None)
                    .await
                    .unwrap();
                assert_eq!(
                    got.rows,
                    case.oracle(&left, &right),
                    "{case:?} batch_rows {batch_rows} group_bytes {group_bytes}"
                );
            }
        }
    }
}

/// One key over many left batches and a large group, without a residual: every right row is
/// moved past once, however many left batches the key spans (the walk carries its place).
#[tokio::test]
async fn a_group_is_read_once_across_left_batches() {
    let left: Vec<Row> = (0..400).map(|i| row(Some(1), Some(i * 5), i)).collect();
    let right: Vec<Row> = (0..1000)
        .map(|i| row(Some(1), Some(i * 2), 1000 + i))
        .collect();
    for backward in [true, false] {
        for group_bytes in [usize::MAX, 4096] {
            let case = Case {
                join_type: JoinType::LeftOuter,
                backward,
                exact: true,
                residual: false,
            };
            let got = run_case(case, &left, &right, 16, group_bytes, None)
                .await
                .unwrap();
            assert_eq!(got.rows, case.oracle(&left, &right), "{case:?}");
            assert!(
                got.right_rows <= right.len() as u64,
                "{case:?}: {} right rows visited for {}",
                got.right_rows,
                right.len()
            );
            if group_bytes != usize::MAX {
                assert!(got.groups_spilled > 0, "{case:?}: the group spilled");
            }
        }
    }
}

/// A residual with a spilled group: results stay the oracle's, memory is released, and the group
/// did spill.
#[tokio::test]
async fn a_residual_over_a_spilled_group() {
    let left: Vec<Row> = (0..300).map(|i| row(Some(1), Some(i * 7), i)).collect();
    let right: Vec<Row> = (0..2000).map(|i| row(Some(1), Some(i), 5000 + i)).collect();
    for backward in [true, false] {
        let case = Case {
            join_type: JoinType::LeftOuter,
            backward,
            exact: false,
            residual: true,
        };
        let got = run_case(case, &left, &right, 32, 2048, None).await.unwrap();
        assert_eq!(got.rows, case.oracle(&left, &right), "{case:?}");
        assert!(got.groups_spilled > 0, "{case:?}: the group spilled");
    }
}

/// Spark's forward scan evaluates its distance on the matched row and on the next passing row,
/// then stops: a raising distance on a later row raises nothing.
#[tokio::test]
async fn the_forward_distance_runs_on_the_match_and_the_next_passing_row() {
    let left = vec![row(Some(1), Some(10), 1)];
    let at = |poison_at: i64| -> Vec<Row> {
        (0..6)
            .map(|i| {
                let t = 9 + i;
                row(
                    Some(1),
                    Some(t),
                    if t == poison_at { POISON } else { 100 + i },
                )
            })
            .collect()
    };
    for residual in [false, true] {
        let case = Case {
            join_type: JoinType::Inner,
            backward: false,
            exact: true,
            residual,
        };
        // Right values 9..=14; left 10: the match is the first passing row (10, or a later one
        // the residual keeps), then the next passing one.
        let passing: Vec<i64> = at(-1)
            .iter()
            .filter(|r| r.t >= Some(10) && (!residual || Residual::eval(1, r.w)))
            .filter_map(|r| r.t)
            .collect();
        for poison_at in 9..=14 {
            let right = at(poison_at);
            // A poisoned row changes the residual's answer for it: recompute what passes.
            let passing_now: Vec<i64> = right
                .iter()
                .filter(|r| r.t >= Some(10) && (!residual || Residual::eval(1, r.w)))
                .filter_map(|r| r.t)
                .collect();
            let raises = passing_now.iter().take(2).any(|&t| t == poison_at);
            for batch_rows in [1, 2, 6] {
                let got = run_case(
                    case,
                    &left,
                    &right,
                    batch_rows,
                    usize::MAX,
                    Some(Arc::new(Poison)),
                )
                .await;
                assert_eq!(
                    got.is_err(),
                    raises,
                    "residual {residual} poison at {poison_at} batch_rows {batch_rows} \
                     (passing {passing:?})"
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn as_of_matches_the_oracle(
        left in prop::collection::vec((0..4i64, prop::option::of(0..40i64), 0..50i64), 0..40),
        right_keys in prop::collection::vec((0..4i64, prop::option::weighted(0.9, 0..40i64)), 0..40),
        case_index in 0..16usize,
        batch_rows in 1..9usize,
        spill in any::<bool>(),
    ) {
        let left: Vec<Row> = left.into_iter().map(|(k, t, w)| row(Some(k), t, w)).collect();
        // Right values unique per key, so the nearest row is one row.
        let mut seen = std::collections::HashSet::new();
        let right: Vec<Row> = right_keys
            .into_iter()
            .enumerate()
            .filter(|(_, (k, t))| t.is_none() || seen.insert((*k, *t)))
            .map(|(i, (k, t))| row(Some(k), t, 1000 + i as i64))
            .collect();
        let case = cases()[case_index];
        let group_bytes = if spill { 64 } else { usize::MAX };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let got = rt
            .block_on(run_case(case, &left, &right, batch_rows, group_bytes, None))
            .unwrap();
        prop_assert_eq!(got.rows, case.oracle(&left, &right), "{:?}", case);
    }
}
