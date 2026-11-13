// SPDX-License-Identifier: Apache-2.0
//! A selection of a batch's rows inside one operator (design P2-E3 U2).
//!
//! An operator the planner fused a filter into ([`RowSelect`]: a predicate
//! and a scan's runtime-filter row stage) evaluates them over each input
//! batch and works on the selected rows by id ([`Selection`]) instead of a
//! filtered copy: a hash probe encodes the keys of those rows only and
//! gathers its output from the input batch by composed ids. A selection
//! never leaves the operator that made it ([`Batch`] has no selection
//! field, ADR 0009).
//!
//! The row stage runs first, over the whole batch, as the scan would (its
//! counters, measurement windows and switch-off see the rows they saw in the
//! scan). The predicate then sees the rows a `Filter` above that scan saw:
//! when the stage dropped at least half of the batch, only the rows it kept
//! (a copy of them, as the scan made), else the whole batch (it cannot
//! raise, so rows the stage dropped change no result). A row is selected when
//! both keep it (a NULL predicate value does not). The copy holds only the
//! columns the predicate reads when the planner compiled it over those
//! ([`Narrow`]).

use std::sync::Arc;

use arrow_array::{Array, BooleanArray, RecordBatch, UInt32Array};
use arrow_buffer::{BooleanBuffer, ScalarBuffer};
use krishiv_types::batch::Batch;

use super::expr::Predicate;
use super::runtime_filter::{SharedStage, scatter};
use crate::error::{Result, internal};

/// The selected rows of one batch: ascending row ids, each below the
/// batch's row count. Empty selects no row; "every row" is `None` where a
/// selection is optional.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Selection {
    rows: ScalarBuffer<u32>,
}

impl Selection {
    /// The rows `mask` sets (NULL is not set).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a mask longer than `u32` rows.
    pub(crate) fn of_mask(mask: &BooleanArray) -> Result<Self> {
        u32::try_from(mask.len()).map_err(|_| internal("a selection over 2^32 rows".to_owned()))?;
        let rows: Vec<u32> = match mask.nulls() {
            Some(n) => (mask.values() & n.inner()).set_indices_u32().collect(),
            None => mask.values().set_indices_u32().collect(),
        };
        Ok(Self { rows: rows.into() })
    }

    /// How many rows are selected.
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether no row is selected.
    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The batch row of the `i`-th selected row.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for `i` at or past [`Self::len`].
    pub(crate) fn row(&self, i: u32) -> Result<u32> {
        self.rows
            .get(i as usize)
            .copied()
            .ok_or_else(|| internal(format!("row {i} is past a selection of {}", self.len())))
    }

    /// The batch rows of the selected positions `ids`, in their order.
    ///
    /// # Errors
    /// As [`Self::row`].
    pub(crate) fn compose(&self, ids: &[u32]) -> Result<Vec<u32>> {
        ids.iter().map(|&i| self.row(i)).collect()
    }

    /// The selected rows as a `take` index.
    pub(crate) fn array(&self) -> UInt32Array {
        UInt32Array::new(self.rows.clone(), None)
    }
}

/// The row ids of `selection` composed with `ids` (positions in the
/// selection), or `ids` themselves without a selection.
///
/// # Errors
/// As [`Selection::row`].
pub(crate) fn batch_rows(selection: Option<&Selection>, ids: &[u32]) -> Result<UInt32Array> {
    Ok(match selection {
        Some(s) => UInt32Array::from(s.compose(ids)?),
        None => UInt32Array::from(ids.to_vec()),
    })
}

/// A fused predicate compiled over only the columns it reads: column
/// `columns[i]` of the batches received is its column `i`. The copy of the
/// rows a selective row stage kept holds those columns only (module docs).
#[derive(Debug, Clone)]
pub(crate) struct Narrow {
    /// The columns of the batches received the predicate reads, ascending.
    pub(crate) columns: Vec<usize>,
    /// Those columns' schema.
    pub(crate) schema: arrow_schema::SchemaRef,
    /// The same predicate over those columns.
    pub(crate) predicate: Arc<dyn Predicate>,
}

/// What a fused operator keeps of each input batch: the rows `predicate`
/// is TRUE for and the row stage keeps. One per driver.
#[derive(Debug)]
pub(crate) struct RowSelect {
    predicate: Option<Arc<dyn Predicate>>,
    narrow: Option<Narrow>,
    runtime: Option<SharedStage>,
}

impl RowSelect {
    /// A selection by `predicate` (also compiled as `narrow`, for the copy
    /// of a selective row stage's rows) and the row stage `runtime`.
    pub(crate) fn new(
        predicate: Option<Arc<dyn Predicate>>,
        narrow: Option<Narrow>,
        runtime: Option<SharedStage>,
    ) -> Self {
        Self {
            predicate,
            narrow,
            runtime,
        }
    }

    /// The rows of `batch` kept, as a mask without NULLs (`None`: every
    /// row), evaluated in driver partition `partition`.
    ///
    /// # Errors
    /// The predicate's errors; `INTERNAL_ERROR` for a mask of the wrong
    /// length.
    pub(crate) fn mask(&mut self, batch: &Batch, partition: u32) -> Result<Option<BooleanArray>> {
        let n = batch.num_rows();
        let runtime = match &self.runtime {
            Some(stage) if n > 0 => stage.mask(batch.data())?.map(|m| m.values().clone()),
            _ => None,
        };
        let all = predicate_after(
            self.predicate.as_deref(),
            self.narrow.as_ref(),
            batch.data(),
            partition,
            runtime,
        )?;
        Ok(all
            .filter(|m| m.count_set_bits() < n)
            .map(|m| BooleanArray::new(m, None)))
    }

    /// The rows of `batch` kept (`None`: every row), evaluated in driver
    /// partition `partition`.
    ///
    /// # Errors
    /// As [`Self::mask`].
    pub(crate) fn select(&mut self, batch: &Batch, partition: u32) -> Result<Option<Selection>> {
        self.mask(batch, partition)?
            .map(|m| Selection::of_mask(&m))
            .transpose()
    }
}

/// The rows of `data` that `predicate` is TRUE for among those the row
/// stage kept (`runtime`, `None`: every row), as a mask over all of `data`
/// (`None`: every row). With a stage that dropped at least half of the rows,
/// the predicate sees only the rows kept (module docs), through `narrow`
/// when there is one.
///
/// # Errors
/// The predicate's errors; `INTERNAL_ERROR` for a mask of the wrong length
/// or a narrow column past the batch.
fn predicate_after(
    predicate: Option<&dyn Predicate>,
    narrow: Option<&Narrow>,
    data: &RecordBatch,
    partition: u32,
    runtime: Option<BooleanBuffer>,
) -> Result<Option<BooleanBuffer>> {
    let Some(predicate) = predicate else {
        return Ok(runtime);
    };
    let n = data.num_rows();
    let truth = |mask: &BooleanArray, rows: usize| -> Result<BooleanBuffer> {
        if mask.len() != rows {
            return Err(internal(
                "a fused filter returned the wrong row count".to_owned(),
            ));
        }
        Ok(match mask.nulls() {
            Some(nulls) => mask.values() & nulls.inner(),
            None => mask.values().clone(),
        })
    };
    let kept_rows = runtime.as_ref().map(BooleanBuffer::count_set_bits);
    match (runtime, kept_rows) {
        (Some(kept), Some(rows)) if rows * 2 <= n => {
            if rows == 0 {
                return Ok(Some(kept));
            }
            let stage = BooleanArray::new(kept.clone(), None);
            let inner = match narrow {
                Some(narrow) => {
                    let picked = narrow
                        .columns
                        .iter()
                        .map(|&c| {
                            data.columns().get(c).cloned().ok_or_else(|| {
                                internal(format!("a fused filter's column {c} is past the batch"))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let columns = RecordBatch::try_new(Arc::clone(&narrow.schema), picked)
                        .map_err(|e| internal(format!("a fused filter's columns: {e}")))?;
                    let copy = arrow_select::filter::filter_record_batch(&columns, &stage)
                        .map_err(|e| internal(format!("a fused row stage's copy: {e}")))?;
                    truth(&narrow.predicate.evaluate_in(&copy, partition)?, rows)?
                }
                None => {
                    let copy = arrow_select::filter::filter_record_batch(data, &stage)
                        .map_err(|e| internal(format!("a fused row stage's copy: {e}")))?;
                    truth(&predicate.evaluate_in(&copy, partition)?, rows)?
                }
            };
            // Spread the kept rows' values back over the batch.
            Ok(Some(scatter(&kept, &inner)))
        }
        (runtime, _) => {
            let p = truth(&predicate.evaluate_in(data, partition)?, n)?;
            Ok(Some(match runtime {
                Some(r) => &p & &r,
                None => p,
            }))
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::Arc;

    use arrow_array::{BooleanArray, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use krishiv_types::batch::Batch;

    use super::{Narrow, RowSelect, Selection, batch_rows, predicate_after};
    use crate::operators::FnPredicate;

    fn batch(v: &[i64]) -> Batch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let data = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(v.to_vec()))]);
        Batch::new(data.unwrap_or_else(|e| unreachable!("{e}")))
    }

    #[test]
    fn a_selection_is_ascending_ids_without_nulls() {
        let mask = BooleanArray::from(vec![Some(true), None, Some(false), Some(true)]);
        let s = Selection::of_mask(&mask).unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(s.array().values().to_vec(), vec![0, 3]);
        assert_eq!(s.compose(&[1, 0]).ok(), Some(vec![3, 0]));
        let rows = |sel, ids: &[u32]| batch_rows(sel, ids).map(|a| a.values().to_vec()).ok();
        assert_eq!(rows(Some(&s), &[1]), Some(vec![3]));
        assert_eq!(rows(None, &[1]), Some(vec![1]));
        // A position past the selection is an error, never a wrong row.
        assert!(s.row(2).is_err());
        assert!(s.compose(&[0, 2]).is_err());
        assert!(!s.is_empty());
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn a_predicate_selects_or_keeps_every_row() {
        let even = FnPredicate(|b: &RecordBatch| {
            let v = arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(
                b.column(0),
            );
            Ok(v.iter().map(|x| x.map(|x| x % 2 == 0)).collect())
        });
        let mut select = RowSelect::new(Some(Arc::new(even)), None, None);
        let s = select.select(&batch(&[1, 2, 3, 4]), 0);
        let s = s.unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(s.map(|s| s.array().values().to_vec()), Some(vec![1, 3]));
        let all = select.select(&batch(&[2, 4]), 0);
        assert_eq!(all.unwrap_or_else(|e| unreachable!("{e}")), None);
        let none = select.select(&batch(&[1]), 0);
        let none = none.unwrap_or_else(|e| unreachable!("{e}"));
        assert!(none.is_some_and(|s| s.is_empty()));
        assert_eq!(
            RowSelect::new(None, None, None)
                .select(&batch(&[1]), 0)
                .ok(),
            Some(None)
        );
    }

    #[test]
    fn the_predicate_sees_only_the_rows_a_selective_row_stage_kept() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use arrow_buffer::BooleanBuffer;

        let seen = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&seen);
        let even = FnPredicate(move |b: &RecordBatch| {
            counted.fetch_add(b.num_rows(), Ordering::Relaxed);
            let v = arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(
                b.column(0),
            );
            Ok(v.iter().map(|x| x.map(|x| x % 2 == 0)).collect())
        });
        let data = batch(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let bits = |b: &[bool]| BooleanBuffer::from(b.to_vec());
        let ids = |m: Option<BooleanBuffer>| m.map(|m| m.set_indices().collect::<Vec<_>>());
        let run = |runtime| predicate_after(Some(&even), None, data.data(), 0, runtime);
        // The stage keeps 2 of 8: the predicate reads those 2 only.
        let sparse = bits(&[false, true, false, false, false, false, true, false]);
        let got = run(Some(sparse)).unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(ids(got), Some(vec![6]));
        assert_eq!(seen.swap(0, Ordering::Relaxed), 2);
        // Keeps 6 of 8: the whole batch, then both masks.
        let dense = bits(&[true, true, false, true, true, true, true, false]);
        let got = run(Some(dense)).unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(ids(got), Some(vec![0, 4, 6]));
        assert_eq!(seen.swap(0, Ordering::Relaxed), 8);
        // Keeps nothing: the predicate is not evaluated.
        let got = run(Some(bits(&[false; 8]))).unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(ids(got), Some(vec![]));
        assert_eq!(seen.load(Ordering::Relaxed), 0);
        // Keeps exactly half (4 of 8): still the kept rows only.
        let half = bits(&[true, true, true, true, false, false, false, false]);
        let got = run(Some(half)).unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(ids(got), Some(vec![0, 2]));
        assert_eq!(seen.swap(0, Ordering::Relaxed), 4);
        // Keeps every row, or there is no row stage: the whole batch.
        let got = run(Some(bits(&[true; 8]))).unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(ids(got), Some(vec![0, 2, 4, 6]));
        assert_eq!(seen.swap(0, Ordering::Relaxed), 8);
        let got = run(None).unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(ids(got), Some(vec![0, 2, 4, 6]));
        assert_eq!(seen.swap(0, Ordering::Relaxed), 8);
    }

    #[test]
    fn the_copy_holds_only_the_columns_a_narrow_predicate_reads() {
        use std::sync::Mutex;

        use arrow_array::StringArray;
        use arrow_buffer::BooleanBuffer;

        // Columns (s, v, t): the predicate reads v only, as column 0 of
        // its narrow form; NULL v is not selected.
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, false),
            Field::new("v", DataType::Int64, true),
            Field::new("t", DataType::Utf8, false),
        ]));
        let s: Vec<String> = (0..8).map(|i| format!("s{i}")).collect();
        let v = Int64Array::from(vec![
            Some(0),
            None,
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            Some(7),
        ]);
        let data = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(s.clone())),
                Arc::new(v),
                Arc::new(StringArray::from(s)),
            ],
        )
        .unwrap_or_else(|e| unreachable!("{e}"));
        let shapes = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&shapes);
        let even_at = |column: usize| {
            move |b: &RecordBatch| {
                let v = arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(
                    b.column(column),
                );
                Ok(v.iter().map(|x| x.map(|x| x % 2 == 0)).collect())
            }
        };
        let full = FnPredicate(even_at(1));
        let narrow_even = even_at(0);
        let narrow = Narrow {
            columns: vec![1],
            schema: Arc::new(
                data.schema()
                    .project(&[1])
                    .unwrap_or_else(|e| unreachable!("{e}")),
            ),
            predicate: Arc::new(FnPredicate(move |b: &RecordBatch| {
                if let Ok(mut s) = seen.lock() {
                    s.push((b.num_columns(), b.num_rows()));
                }
                narrow_even(b)
            })),
        };
        let bits = |b: &[bool]| BooleanBuffer::from(b.to_vec());
        let ids = |m: Option<BooleanBuffer>| m.map(|m| m.set_indices().collect::<Vec<_>>());
        // The stage keeps rows 1 (v NULL), 2 and 5 of 8: the narrow form
        // reads one column of those three rows; only row 2 is selected.
        let kept = bits(&[false, true, true, false, false, true, false, false]);
        let got = predicate_after(Some(&full), Some(&narrow), &data, 0, Some(kept.clone()));
        assert_eq!(
            ids(got.unwrap_or_else(|e| unreachable!("{e}"))),
            Some(vec![2])
        );
        assert_eq!(shapes.lock().map(|s| s.clone()).ok(), Some(vec![(1, 3)]));
        // Without the narrow form the same rows are selected.
        let got = predicate_after(Some(&full), None, &data, 0, Some(kept));
        assert_eq!(
            ids(got.unwrap_or_else(|e| unreachable!("{e}"))),
            Some(vec![2])
        );
        // A dense stage reads the whole batch through the full predicate.
        let dense = bits(&[true, true, true, true, true, false, false, false]);
        let got = predicate_after(Some(&full), Some(&narrow), &data, 0, Some(dense));
        assert_eq!(
            ids(got.unwrap_or_else(|e| unreachable!("{e}"))),
            Some(vec![0, 2, 4])
        );
        assert_eq!(shapes.lock().map(|s| s.len()).ok(), Some(1));
        // A narrow column past the batch is an error, never a wrong row.
        let broken = Narrow {
            columns: vec![3],
            schema: Arc::clone(&narrow.schema),
            predicate: Arc::new(FnPredicate(even_at(0))),
        };
        let sparse = bits(&[true, false, false, false, false, false, false, false]);
        assert!(predicate_after(Some(&full), Some(&broken), &data, 0, Some(sparse)).is_err());
    }
}
