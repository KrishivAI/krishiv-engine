// SPDX-License-Identifier: Apache-2.0
//! Lazy conditional evaluation: `CASE WHEN`, `IF`, `COALESCE` and the
//! short-circuiting `AND`/`OR` evaluate a branch only on the rows that reach
//! it, so an error in a branch no row takes (an ANSI division by zero, an
//! overflow) never surfaces, exactly as in Spark.
//!
//! The control flow is mask -> filter -> evaluate -> zip: a condition is
//! evaluated on the rows still undecided, the rows it selects ([`Selection`])
//! are handed to the branch's value, and the per-branch results are zipped
//! back into one column in row order. Operands are [`Lazy`] producers:
//! given a [`Selection`] they return one value per selected row. A producer
//! over batch columns typically [`Selection::gather`]s them first and runs a
//! kernel on the gathered rows; nested conditionals simply pass the selection
//! they were given on to their own branches.
//!
//! Semantics (Spark's):
//! - `CASE WHEN c1 THEN v1 ... [ELSE e] END`: the first condition that is
//!   true wins; a NULL condition is not true; no `ELSE` gives NULL.
//! - `COALESCE(a, b, ...)`: the first non-NULL argument; later arguments are
//!   evaluated only on the rows where every earlier one is NULL.
//! - `a AND b`: false when `a` is false, without evaluating `b` there; NULL
//!   `AND` false is false, NULL `AND` true or NULL is NULL. `OR` is the dual.

use std::fmt;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, UInt32Array, new_null_array};
use arrow_schema::DataType;

use crate::error::KernelError;

/// The rows of a batch a producer must evaluate: ascending, distinct row
/// indices into the batch the conditional runs over.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Selection {
    rows: Vec<u32>,
}

impl Selection {
    /// Every row of a batch of `rows` rows.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the batch has more than `u32::MAX` rows (batches
    /// are thousands of rows); `UNABLE_TO_ACQUIRE_MEMORY` if the index list
    /// cannot be allocated.
    pub fn all(rows: usize) -> Result<Self, KernelError> {
        let n = u32::try_from(rows)
            .map_err(|_| KernelError::internal(format!("a batch of {rows} rows")))?;
        let mut v = crate::error::with_capacity::<u32>(rows)?;
        v.extend(0..n);
        Ok(Self { rows: v })
    }

    /// The rows at `indices`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` unless `indices` is strictly ascending.
    pub fn from_indices(indices: Vec<u32>) -> Result<Self, KernelError> {
        if indices.windows(2).any(|w| w[0] >= w[1]) {
            return Err(KernelError::internal(
                "a selection must be strictly ascending",
            ));
        }
        Ok(Self { rows: indices })
    }

    /// The number of selected rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether no row is selected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The selected row indices, ascending.
    #[must_use]
    pub fn rows(&self) -> &[u32] {
        &self.rows
    }

    /// `column` restricted to the selected rows, in order.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a selected row is past the end of `column`.
    pub fn gather(&self, column: &dyn Array) -> Result<ArrayRef, KernelError> {
        let indices = UInt32Array::from(self.rows.clone());
        // `take` panics on an index past the end unless asked to check.
        let options = arrow_select::take::TakeOptions { check_bounds: true };
        arrow_select::take::take(column, &indices, Some(options))
            .map_err(|e| KernelError::internal(e.to_string()))
    }
}

/// A value computed on demand: one entry per row of the [`Selection`] it is
/// given, in selection order, and no work for any other row.
pub trait Lazy {
    /// The values of the selected rows.
    ///
    /// # Errors
    /// Any kernel error raised by a selected row.
    fn eval(&mut self, rows: &Selection) -> Result<ArrayRef, KernelError>;
}

impl<F> Lazy for F
where
    F: FnMut(&Selection) -> Result<ArrayRef, KernelError>,
{
    fn eval(&mut self, rows: &Selection) -> Result<ArrayRef, KernelError> {
        self(rows)
    }
}

/// One `WHEN condition THEN value` of a `CASE`.
pub struct When<'a> {
    /// A `boolean` producer.
    pub condition: &'a mut dyn Lazy,
    /// The branch's value.
    pub value: &'a mut dyn Lazy,
}

impl fmt::Debug for When<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("When")
    }
}

/// The rows still undecided: their indices in the batch and their positions
/// in the selection the conditional was asked for.
struct Live {
    rows: Selection,
    pos: Vec<u32>,
}

impl Live {
    fn root(rows: &Selection) -> Result<Self, KernelError> {
        Ok(Self {
            pos: Selection::all(rows.len())?.rows,
            rows: rows.clone(),
        })
    }

    fn empty() -> Self {
        Self {
            rows: Selection::default(),
            pos: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.rows.len()
    }

    /// Splits into the rows where `pick(i)` holds (`i` indexes `self`) and
    /// the rest, both in order.
    fn split(&self, pick: impl Fn(usize) -> bool) -> (Self, Self) {
        let (mut yes, mut no) = (Self::empty(), Self::empty());
        for (i, (&row, &pos)) in self.rows.rows.iter().zip(&self.pos).enumerate() {
            let side = if pick(i) { &mut yes } else { &mut no };
            side.rows.rows.push(row);
            side.pos.push(pos);
        }
        (yes, no)
    }
}

/// Collects the per-branch results and zips them into one column.
struct Zip {
    sources: Vec<ArrayRef>,
    /// Per root position: `(source, index in source)`, or `NONE` for NULL.
    place: Vec<(u32, u32)>,
}

const NONE: (u32, u32) = (u32::MAX, u32::MAX);

impl Zip {
    fn new(rows: usize) -> Result<Self, KernelError> {
        let mut place = crate::error::with_capacity::<(u32, u32)>(rows)?;
        place.resize(rows, NONE);
        Ok(Self {
            sources: Vec::new(),
            place,
        })
    }

    /// Puts `source[at]` at root position `to` for every `(to, at)`.
    fn put(
        &mut self,
        source: ArrayRef,
        entries: impl Iterator<Item = (u32, u32)>,
    ) -> Result<(), KernelError> {
        let id = u32::try_from(self.sources.len())
            .map_err(|_| KernelError::internal("too many conditional branches"))?;
        for (to, at) in entries {
            let slot = self
                .place
                .get_mut(usize::try_from(to).unwrap_or(usize::MAX))
                .ok_or_else(|| KernelError::internal("a conditional placed a row twice"))?;
            *slot = (id, at);
        }
        self.sources.push(source);
        Ok(())
    }

    /// Puts all of `source`, produced for exactly the rows of `live`.
    fn put_all(&mut self, live: &Live, source: ArrayRef) -> Result<(), KernelError> {
        expect_rows(&source, live.len())?;
        let entries = live.pos.iter().copied().zip(0..);
        self.put(source, entries)
    }

    fn finish(self, ty: &DataType) -> Result<ArrayRef, KernelError> {
        let identity = |s: &ArrayRef| {
            s.len() == self.place.len()
                && self
                    .place
                    .iter()
                    .zip(0_u32..)
                    .all(|(&(src, at), i)| src == 0 && at == i)
        };
        if let [only] = self.sources.as_slice()
            && identity(only)
        {
            return Ok(Arc::clone(only));
        }
        let nulls = new_null_array(ty, 1);
        let null_id = self.sources.len();
        let mut refs: Vec<&dyn Array> = self.sources.iter().map(AsRef::as_ref).collect();
        refs.push(nulls.as_ref());
        let indices: Vec<(usize, usize)> = self
            .place
            .iter()
            .map(|&(src, at)| {
                if (src, at) == NONE {
                    (null_id, 0)
                } else {
                    (
                        usize::try_from(src).unwrap_or(null_id),
                        usize::try_from(at).unwrap_or(0),
                    )
                }
            })
            .collect();
        arrow_select::interleave::interleave(&refs, &indices)
            .map_err(|e| KernelError::internal(e.to_string()))
    }
}

fn expect_rows(a: &ArrayRef, rows: usize) -> Result<(), KernelError> {
    if a.len() == rows {
        Ok(())
    } else {
        Err(KernelError::internal(format!(
            "a conditional operand returned {} values for {rows} rows",
            a.len()
        )))
    }
}

/// A branch value as the conditional's type: a `NULL` literal takes it, any
/// other type must already be it.
fn typed(a: ArrayRef, ty: &DataType) -> Result<ArrayRef, KernelError> {
    if a.data_type() == ty {
        Ok(a)
    } else if a.data_type() == &DataType::Null {
        Ok(new_null_array(ty, a.len()))
    } else {
        Err(KernelError::internal(format!(
            "a conditional branch of type {} where {ty} is expected",
            a.data_type()
        )))
    }
}

fn boolean(a: &ArrayRef, rows: usize) -> Result<BooleanArray, KernelError> {
    expect_rows(a, rows)?;
    match a.data_type() {
        DataType::Boolean => Ok(a.as_boolean().clone()),
        DataType::Null => Ok(BooleanArray::new_null(rows)),
        other => Err(KernelError::internal(format!(
            "a condition of type {other}"
        ))),
    }
}

/// `CASE WHEN ... [ELSE ...] END` over `rows`, of type `ty`.
///
/// Each condition runs on the rows no earlier condition selected; each value
/// runs on the rows its condition selected; `otherwise` runs on what is left
/// (without it those rows are NULL). A row whose condition is NULL or false
/// moves on to the next branch.
///
/// # Errors
/// Any error a producer raises for a row it was given; `INTERNAL_ERROR` for
/// a producer that returns the wrong number of values or a value of another
/// type than `ty`.
pub fn case_when(
    rows: &Selection,
    branches: &mut [When<'_>],
    otherwise: Option<&mut dyn Lazy>,
    ty: &DataType,
) -> Result<ArrayRef, KernelError> {
    let mut zip = Zip::new(rows.len())?;
    let mut live = Live::root(rows)?;
    for branch in branches {
        if live.len() == 0 {
            break;
        }
        let cond = boolean(&branch.condition.eval(&live.rows)?, live.len())?;
        let (hit, rest) = live.split(|i| cond.is_valid(i) && cond.value(i));
        if hit.len() > 0 {
            let value = typed(branch.value.eval(&hit.rows)?, ty)?;
            zip.put_all(&hit, value)?;
        }
        live = rest;
    }
    if let Some(otherwise) = otherwise
        && live.len() > 0
    {
        let value = typed(otherwise.eval(&live.rows)?, ty)?;
        zip.put_all(&live, value)?;
    }
    zip.finish(ty)
}

/// `IF(condition, then, otherwise)`: a one-branch [`case_when`].
///
/// # Errors
/// As [`case_when`].
pub fn if_else(
    rows: &Selection,
    condition: &mut dyn Lazy,
    then: &mut dyn Lazy,
    otherwise: &mut dyn Lazy,
    ty: &DataType,
) -> Result<ArrayRef, KernelError> {
    let mut when = [When {
        condition,
        value: then,
    }];
    case_when(rows, &mut when, Some(otherwise), ty)
}

/// `COALESCE(args...)` over `rows`, of type `ty`: each argument runs only on
/// the rows every earlier argument left NULL; the last argument's NULLs stay.
///
/// # Errors
/// As [`case_when`].
pub fn coalesce(
    rows: &Selection,
    args: &mut [&mut dyn Lazy],
    ty: &DataType,
) -> Result<ArrayRef, KernelError> {
    let mut zip = Zip::new(rows.len())?;
    let mut live = Live::root(rows)?;
    let last = args.len().saturating_sub(1);
    for (k, arg) in args.iter_mut().enumerate() {
        if live.len() == 0 {
            break;
        }
        let value = typed(arg.eval(&live.rows)?, ty)?;
        expect_rows(&value, live.len())?;
        if k == last {
            zip.put_all(&live, value)?;
            live = Live::empty();
        } else {
            let (rest, _) = live.split(|i| value.is_null(i));
            // The source index of a row is its position in the value just computed.
            let entries = live
                .pos
                .iter()
                .copied()
                .zip(0_u32..)
                .filter(|&(_, at)| value.is_valid(usize::try_from(at).unwrap_or(0)));
            zip.put(Arc::clone(&value), entries)?;
            live = rest;
        }
    }
    zip.finish(ty)
}

/// `left AND right` over `rows`: `right` runs only where `left` is not false.
///
/// # Errors
/// As [`case_when`].
pub fn and(
    rows: &Selection,
    left: &mut dyn Lazy,
    right: &mut dyn Lazy,
) -> Result<BooleanArray, KernelError> {
    connective(rows, left, right, false)
}

/// `left OR right` over `rows`: `right` runs only where `left` is not true.
///
/// # Errors
/// As [`case_when`].
pub fn or(
    rows: &Selection,
    left: &mut dyn Lazy,
    right: &mut dyn Lazy,
) -> Result<BooleanArray, KernelError> {
    connective(rows, left, right, true)
}

/// `AND` (`dominant` = false) or `OR` (`dominant` = true): a left operand
/// equal to `dominant` decides the row; otherwise the right one is needed.
fn connective(
    rows: &Selection,
    left: &mut dyn Lazy,
    right: &mut dyn Lazy,
    dominant: bool,
) -> Result<BooleanArray, KernelError> {
    let l = boolean(&left.eval(rows)?, rows.len())?;
    let live = Live::root(rows)?;
    let (_, undecided) = live.split(|i| l.is_valid(i) && l.value(i) == dominant);
    let r = if undecided.len() > 0 {
        boolean(&right.eval(&undecided.rows)?, undecided.len())?
    } else {
        BooleanArray::new_null(0)
    };
    // Row `p` of `rows`: dominant if decided; else the right operand, with a
    // NULL left operand making a non-dominant right operand NULL.
    let mut out: Vec<Option<bool>> = vec![Some(dominant); rows.len()];
    for (j, &p) in undecided.pos.iter().enumerate() {
        let p = usize::try_from(p).unwrap_or(0);
        let left_null = l.is_null(p);
        out[p] = if r.is_null(j) {
            None
        } else if r.value(j) == dominant {
            Some(dominant)
        } else if left_null {
            None
        } else {
            Some(!dominant)
        };
    }
    Ok(out.into_iter().collect())
}
