// SPDX-License-Identifier: Apache-2.0
//! [`super::conjunction`]: Kleene answers in any order, barriers over exactly the rows the
//! conjuncts written before them leave open, and the measured order (study 60 (d) part 2).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};

use super::conjunction::{Part, conjunction, order_of};
use super::cost::Cost;
use super::{Column, PhysExpr, Value};
use crate::error::Result;

/// A boolean column that counts the rows it is evaluated over.
#[derive(Debug)]
struct Counted {
    inner: Column,
    rows: AtomicUsize,
}

impl Counted {
    fn new(index: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Column {
                index,
                data_type: ArrowType::Boolean,
                nullable: true,
            },
            rows: AtomicUsize::new(0),
        })
    }

    fn take(&self) -> usize {
        self.rows.swap(0, Ordering::Relaxed)
    }
}

impl PhysExpr for Counted {
    fn data_type(&self) -> &ArrowType {
        self.inner.data_type()
    }

    fn nullable(&self) -> bool {
        true
    }

    fn eval(&self, batch: &RecordBatch, partition: u32) -> Result<Value> {
        self.rows.fetch_add(batch.num_rows(), Ordering::Relaxed);
        self.inner.eval(batch, partition)
    }
}

fn batch(columns: Vec<Vec<Option<bool>>>) -> RecordBatch {
    let fields: Vec<Field> = (0..columns.len())
        .map(|i| Field::new(format!("c{i}"), ArrowType::Boolean, true))
        .collect();
    let arrays: Vec<ArrayRef> = columns
        .into_iter()
        .map(|c| Arc::new(BooleanArray::from(c)) as ArrayRef)
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn answers(e: &dyn PhysExpr, batch: &RecordBatch) -> Vec<Option<bool>> {
    let v = e
        .eval(batch, 0)
        .unwrap()
        .into_array(batch.num_rows())
        .unwrap();
    v.as_boolean().iter().collect()
}

fn kleene_and(values: &[Option<bool>]) -> Option<bool> {
    if values.contains(&Some(false)) {
        Some(false)
    } else if values.contains(&None) {
        None
    } else {
        Some(true)
    }
}

/// Every triple of TRUE, FALSE and NULL, `copies` times over.
fn triples(copies: usize) -> RecordBatch {
    let values = [Some(true), Some(false), None];
    let column = |div: usize| -> Vec<Option<bool>> {
        (0..27 * copies)
            .map(|i| values[(i % 27) / div % 3])
            .collect()
    };
    batch(vec![column(1), column(3), column(9)])
}

/// Kleene answers for every mix of movable conjuncts and barriers and every cost class, before
/// and after the order is fixed; a barrier sees exactly the rows the conjuncts written before it
/// do not make FALSE.
#[test]
fn answers_are_kleene_and_barriers_see_the_open_rows_in_both_phases() {
    let data = triples(100); // 2,700 rows: the order is fixed after 25 batches
    let columns: Vec<Vec<Option<bool>>> = (0..3)
        .map(|c| data.column(c).as_boolean().iter().collect())
        .collect();
    let expected: Vec<Option<bool>> = (0..data.num_rows())
        .map(|r| kleene_and(&[columns[0][r], columns[1][r], columns[2][r]]))
        .collect();
    for movable_bits in 0..8_u8 {
        for cost in [Cost::Cheap, Cost::String, Cost::Costly] {
            let counted: Vec<Arc<Counted>> = (0..3).map(Counted::new).collect();
            let parts = counted
                .iter()
                .enumerate()
                .map(|(i, c)| Part {
                    expr: Arc::clone(c) as Arc<dyn PhysExpr>,
                    movable: movable_bits & (1 << i) != 0,
                    cost,
                })
                .collect();
            let e = conjunction(parts);
            for round in 0..40 {
                assert_eq!(
                    answers(e.as_ref(), &data),
                    expected,
                    "movable={movable_bits:03b} {cost:?} round={round}"
                );
                for (i, c) in counted.iter().enumerate() {
                    let seen = c.take();
                    if movable_bits & (1 << i) == 0 {
                        let open = (0..data.num_rows())
                            .filter(|&r| (0..i).all(|j| columns[j][r] != Some(false)))
                            .count();
                        assert_eq!(
                            seen, open,
                            "barrier {i} movable={movable_bits:03b} {cost:?} round={round}"
                        );
                    }
                }
            }
        }
    }
}

/// After the measured rows, the costly conjunct that keeps the fewest rows runs first over every
/// row, and the other runs over the few rows it leaves; a barrier between them keeps them apart.
#[test]
fn the_most_selective_costly_conjunct_runs_first_once_measured() {
    let rows = 4_096;
    let keeps_all: Vec<Option<bool>> = vec![Some(true); rows];
    let keeps_few: Vec<Option<bool>> = (0..rows).map(|r| Some(r % 100 == 0)).collect();
    let data = batch(vec![keeps_all.clone(), keeps_few, keeps_all]);
    let few = (0..rows).filter(|r| r % 100 == 0).count();
    for barrier_between in [false, true] {
        let counted: Vec<Arc<Counted>> = (0..3).map(Counted::new).collect();
        // Written: keeps-all, then (optionally) a barrier, then keeps-few.
        let order: Vec<usize> = if barrier_between {
            vec![0, 2, 1]
        } else {
            vec![0, 1]
        };
        let parts = order
            .iter()
            .map(|&c| Part {
                expr: Arc::clone(&counted[c]) as Arc<dyn PhysExpr>,
                movable: c != 2,
                cost: Cost::Costly,
            })
            .collect();
        let e = conjunction(parts);
        // 16 batches of 4,096 rows measure; the 17th runs in the measured order.
        for _ in 0..16 {
            answers(e.as_ref(), &data);
        }
        counted.iter().for_each(|c| {
            c.take();
        });
        let out = answers(e.as_ref(), &data);
        assert_eq!(out.iter().filter(|v| **v == Some(true)).count(), few);
        assert_eq!(counted[1].take(), rows, "barrier={barrier_between}");
        let first = if barrier_between { rows } else { few };
        assert_eq!(counted[0].take(), first, "barrier={barrier_between}");
    }
}

/// Literal conjuncts give one value for the batch.
#[test]
fn a_conjunction_of_literals_is_one_value() {
    let data = triples(1);
    let lit = |v: bool| -> Arc<dyn PhysExpr> {
        Arc::new(super::Lit::new(&krishiv_plan::Literal::Boolean(v)).unwrap())
    };
    for (a, b) in [(true, true), (true, false), (false, true)] {
        let e = conjunction(vec![
            Part {
                expr: lit(a),
                movable: true,
                cost: Cost::Cheap,
            },
            Part {
                expr: lit(b),
                movable: true,
                cost: Cost::Cheap,
            },
        ]);
        let v = e.eval(&data, 0).unwrap();
        assert!(v.is_scalar(), "{a} AND {b}");
        assert_eq!(answers(e.as_ref(), &data), vec![Some(a && b); 27]);
    }
}

fn part(column: usize, movable: bool, cost: Cost) -> Part {
    Part {
        expr: Counted::new(column) as Arc<dyn PhysExpr>,
        movable,
        cost,
    }
}

/// The order of a snapshot of measures: each run between barriers by cost class, then kept
/// fraction, ties and unmeasured conjuncts (fraction 1) in written order; barriers stay put.
#[test]
fn the_order_sorts_each_run_by_cost_then_kept_fraction() {
    let parts = vec![
        part(0, true, Cost::Costly), // 0: kept 1/2
        part(0, true, Cost::Cheap),  // 1: kept 9/10
        part(0, true, Cost::Costly), // 2: kept 1/10
        part(0, false, Cost::Cheap), // 3: barrier
        part(0, true, Cost::String), // 4: unmeasured
        part(0, true, Cost::String), // 5: kept 1/4
        part(0, true, Cost::String), // 6: kept 2/8, a tie with 5
    ];
    let measures = [(1, 2), (9, 10), (1, 10), (0, 0), (0, 0), (1, 4), (2, 8)];
    assert_eq!(order_of(&parts, &measures), vec![1, 2, 0, 3, 5, 6, 4]);
}

/// A long run of movable conjuncts (more than the 20 an insertion sort handles) whose order is
/// fixed while other drivers still add to its measures: every driver's answers stay Kleene and
/// the sort sees one consistent snapshot (a comparator over live measures may panic).
#[test]
fn a_long_run_is_ordered_while_other_drivers_measure() {
    let width = 32;
    let rows = 1_024;
    let columns: Vec<Vec<Option<bool>>> = (0..width)
        .map(|c| {
            (0..rows)
                .map(|r| Some((r * 7 + c * 13) % (c + 2) != 0))
                .collect()
        })
        .collect();
    let expected: Vec<Option<bool>> = (0..rows)
        .map(|r| kleene_and(&columns.iter().map(|c| c[r]).collect::<Vec<_>>()))
        .collect();
    let data = batch(columns);
    for _ in 0..8 {
        let parts = (0..width).map(|c| part(c, true, Cost::Costly)).collect();
        let e = conjunction(parts);
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..40 {
                        assert_eq!(answers(e.as_ref(), &data), expected);
                    }
                });
            }
        });
    }
}

/// A movable conjunct that keeps few rows, written after one that keeps most, moves in front of
/// it once measured; the barrier after them still sees exactly the rows neither makes FALSE, and
/// the answers stay Kleene with NULLs in both.
#[test]
fn a_barrier_sees_the_open_rows_after_a_real_reorder() {
    let rows = 4_096;
    let most: Vec<Option<bool>> = (0..rows)
        .map(|r| match r % 10 {
            0 => None,
            1 => Some(false),
            _ => Some(true),
        })
        .collect();
    let few: Vec<Option<bool>> = (0..rows)
        .map(|r| match r % 50 {
            0 | 3 => Some(true),
            7 => None,
            _ => Some(false),
        })
        .collect();
    let other: Vec<Option<bool>> = (0..rows).map(|r| (r % 3 != 0).then_some(true)).collect();
    let expected: Vec<Option<bool>> = (0..rows)
        .map(|r| kleene_and(&[most[r], few[r], other[r]]))
        .collect();
    let open = (0..rows)
        .filter(|&r| most[r] != Some(false) && few[r] != Some(false))
        .count();
    let data = batch(vec![most, few, other]);
    let counted: Vec<Arc<Counted>> = (0..3).map(Counted::new).collect();
    let parts = counted
        .iter()
        .enumerate()
        .map(|(i, c)| Part {
            expr: Arc::clone(c) as Arc<dyn PhysExpr>,
            movable: i != 2,
            cost: Cost::Costly,
        })
        .collect();
    let e = conjunction(parts);
    for _ in 0..16 {
        assert_eq!(answers(e.as_ref(), &data), expected);
    }
    counted.iter().for_each(|c| {
        c.take();
    });
    assert_eq!(answers(e.as_ref(), &data), expected);
    // Reordered: `few` first over every row, then `most` over the few rows it leaves open.
    assert_eq!(counted[1].take(), rows);
    let after_few = data
        .column(1)
        .as_boolean()
        .iter()
        .filter(|v| *v != Some(false))
        .count();
    assert_eq!(counted[0].take(), after_few);
    assert_eq!(counted[2].take(), open);
}
