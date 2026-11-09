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
//! The predicate runs over the whole batch first, as a `Filter` would, and
//! the row stage over the whole batch too, as the scan would (its counters,
//! measurement windows and switch-off see the rows they saw in the scan);
//! a row is selected when both keep it (a NULL predicate value does not).

use std::sync::Arc;

use arrow_array::{Array, BooleanArray, UInt32Array};
use krishiv_types::batch::Batch;

use super::expr::Predicate;
use super::runtime_filter::SharedStage;
use crate::error::{Result, internal};

/// The selected rows of one batch: ascending row ids, each below the
/// batch's row count. Empty selects no row; "every row" is `None` where a
/// selection is optional.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Selection {
    rows: Vec<u32>,
}

impl Selection {
    /// The rows `mask` sets (NULL is not set).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a mask longer than `u32` rows.
    pub(crate) fn of_mask(mask: &BooleanArray) -> Result<Self> {
        let rows = match mask.nulls() {
            Some(n) => (mask.values() & n.inner())
                .set_indices()
                .collect::<Vec<_>>(),
            None => mask.values().set_indices().collect(),
        };
        let rows = rows
            .into_iter()
            .map(|i| {
                u32::try_from(i).map_err(|_| internal("a selection over 2^32 rows".to_owned()))
            })
            .collect::<Result<_>>()?;
        Ok(Self { rows })
    }

    /// How many rows are selected.
    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether no row is selected.
    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The batch row of the `i`-th selected row (`i` itself past the end:
    /// callers index within [`Self::len`]).
    pub(crate) fn row(&self, i: u32) -> u32 {
        self.rows.get(i as usize).copied().unwrap_or(i)
    }

    /// The batch rows of the selected positions `ids`, in their order.
    pub(crate) fn compose(&self, ids: &[u32]) -> Vec<u32> {
        ids.iter().map(|&i| self.row(i)).collect()
    }

    /// The selected rows as a `take` index.
    pub(crate) fn array(&self) -> UInt32Array {
        UInt32Array::from(self.rows.clone())
    }
}

/// The row ids of `selection` composed with `ids` (positions in the
/// selection), or `ids` themselves without a selection.
pub(crate) fn batch_rows(selection: Option<&Selection>, ids: &[u32]) -> UInt32Array {
    match selection {
        Some(s) => UInt32Array::from(s.compose(ids)),
        None => UInt32Array::from(ids.to_vec()),
    }
}

/// What a fused operator keeps of each input batch: the rows `predicate`
/// is TRUE for and the row stage keeps. One per driver.
#[derive(Debug)]
pub(crate) struct RowSelect {
    predicate: Option<Arc<dyn Predicate>>,
    runtime: Option<SharedStage>,
}

impl RowSelect {
    /// A selection by `predicate` and the row stage `runtime`.
    pub(crate) fn new(predicate: Option<Arc<dyn Predicate>>, runtime: Option<SharedStage>) -> Self {
        Self { predicate, runtime }
    }

    /// The rows of `batch` kept, as a mask without NULLs (`None`: every
    /// row), evaluated in driver partition `partition`.
    ///
    /// # Errors
    /// The predicate's errors; `INTERNAL_ERROR` for a mask of the wrong
    /// length.
    pub(crate) fn mask(&mut self, batch: &Batch, partition: u32) -> Result<Option<BooleanArray>> {
        let n = batch.num_rows();
        let predicate = match &self.predicate {
            Some(p) => {
                let mask = p.evaluate_in(batch.data(), partition)?;
                if mask.len() != n {
                    return Err(internal(
                        "a fused filter returned the wrong row count".to_owned(),
                    ));
                }
                Some(match mask.nulls() {
                    Some(nulls) => mask.values() & nulls.inner(),
                    None => mask.values().clone(),
                })
            }
            None => None,
        };
        let runtime = match &self.runtime {
            Some(stage) if n > 0 => stage.mask(batch.data())?.map(|m| m.values().clone()),
            _ => None,
        };
        let all = match (predicate, runtime) {
            (Some(p), Some(r)) => Some(&p & &r),
            (p, r) => p.or(r),
        };
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

#[cfg(all(test, not(loom)))]
mod tests {
    use std::sync::Arc;

    use arrow_array::{BooleanArray, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use krishiv_types::batch::Batch;

    use super::{RowSelect, Selection, batch_rows};
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
        assert_eq!(s.compose(&[1, 0]), vec![3, 0]);
        assert_eq!(batch_rows(Some(&s), &[1]).values().to_vec(), vec![3]);
        assert_eq!(batch_rows(None, &[1]).values().to_vec(), vec![1]);
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
        let mut select = RowSelect::new(Some(Arc::new(even)), None);
        let s = select.select(&batch(&[1, 2, 3, 4]), 0);
        let s = s.unwrap_or_else(|e| unreachable!("{e}"));
        assert_eq!(s.map(|s| s.array().values().to_vec()), Some(vec![1, 3]));
        let all = select.select(&batch(&[2, 4]), 0);
        assert_eq!(all.unwrap_or_else(|e| unreachable!("{e}")), None);
        let none = select.select(&batch(&[1]), 0);
        let none = none.unwrap_or_else(|e| unreachable!("{e}"));
        assert!(none.is_some_and(|s| s.is_empty()));
        assert_eq!(
            RowSelect::new(None, None).select(&batch(&[1]), 0).ok(),
            Some(None)
        );
    }
}
