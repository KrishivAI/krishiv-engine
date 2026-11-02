// SPDX-License-Identifier: Apache-2.0
//! A Top-N bound (design P2-T163, P2-T063): the key of the last row a top-N
//! keeps. A row past it in the sort's order can't enter the top-N, so a scan
//! below may drop it. Rows equal to the bound pass, because a tie may still
//! be kept.
//!
//! Values compare in the sort's total order, not IEEE order: NaN is the
//! greatest float and `-0.0` is `0.0`, which key normalization already gives.
//! NULLs rank by `NULLS FIRST/LAST` alone, so they are the filter's
//! [`nulls_pass`](KeyFilter::nulls_pass), not part of the comparison.

use arrow_array::Array;
use arrow_buffer::BooleanBuffer;
use arrow_schema::DataType;

use super::KeyFilter;
use super::key::{KeyValue, Norm, NormalizedKeys, float_bits, for_each_bytes, normalize_key_bytes};
use super::member::Member;
use crate::request::CmpOp;

/// A bound on non-null keys.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TopNBound {
    /// Never NaN: no float compares past a NaN bound in a useful way, so
    /// [`KeyFilter::top_n_bound`] makes no filter of one.
    pub(crate) value: KeyValue,
    /// `DESC`: rows at or above `value` pass, otherwise at or below.
    pub(crate) descending: bool,
}

/// The float a [`KeyValue::Fixed`] of a float key holds.
fn float_of(bits: i128, single: bool) -> f64 {
    if single {
        u32::try_from(bits).map_or(f64::NAN, |b| f64::from(f32::from_bits(b)))
    } else {
        u64::try_from(bits).map_or(f64::NAN, f64::from_bits)
    }
}

impl TopNBound {
    /// Which rows of `keys` pass. A NULL row's answer is meaningless: the
    /// caller decides those.
    pub(crate) fn pass(&self, keys: &NormalizedKeys) -> BooleanBuffer {
        let n = keys.len();
        let desc = self.descending;
        match (&keys.norm, &self.value) {
            (Norm::Int { values, .. }, KeyValue::Fixed(x)) => BooleanBuffer::collect_bool(n, |i| {
                values.get(i).is_some_and(|&v| {
                    let v = i128::from(v);
                    if desc { v >= *x } else { v <= *x }
                })
            }),
            (Norm::Decimal(values), KeyValue::Fixed(x)) => BooleanBuffer::collect_bool(n, |i| {
                values
                    .get(i)
                    .is_some_and(|v| if desc { v >= x } else { v <= x })
            }),
            (Norm::Float { values, single }, KeyValue::Fixed(x)) => {
                let x = float_of(*x, *single);
                BooleanBuffer::collect_bool(n, |i| {
                    values
                        .get(i)
                        .is_some_and(|&v| if desc { v.is_nan() || v >= x } else { v <= x })
                })
            }
            (Norm::Bytes(a), KeyValue::Bytes(x)) => {
                let mut pass = vec![false; n];
                for_each_bytes(a.as_ref(), |i, b| {
                    if let Some(p) = pass.get_mut(i) {
                        *p = if desc { b >= &**x } else { b <= &**x };
                    }
                });
                BooleanBuffer::collect_bool(n, |i| pass.get(i).copied().unwrap_or(false))
            }
            // Not a bound of this key type: nothing is known, so all pass.
            _ => BooleanBuffer::new_set(n),
        }
    }

    /// Whether `self` drops every row `other` drops, and more. Bounds of one
    /// filter have one key type and direction.
    fn tighter_than(&self, other: &Self, float: Option<bool>) -> bool {
        let ordering = match (&self.value, &other.value, float) {
            (KeyValue::Fixed(a), KeyValue::Fixed(b), Some(single)) => {
                float_of(*a, single).total_cmp(&float_of(*b, single))
            }
            (a, b, _) => a.cmp(b),
        };
        if self.descending {
            ordering.is_gt()
        } else {
            ordering.is_lt()
        }
    }

    /// The comparison statistics may prune with, or `None` when they may
    /// not: statistics leave NaN out, and a `DESC` bound must keep NaN.
    pub(crate) fn comparison(&self, key_type: &DataType) -> Option<(KeyValue, CmpOp)> {
        if self.descending && matches!(key_type, DataType::Float32 | DataType::Float64) {
            return None;
        }
        let op = if self.descending {
            CmpOp::GtEq
        } else {
            CmpOp::LtEq
        };
        Some((self.value.clone(), op))
    }
}

impl KeyFilter {
    /// The filter a top-N publishes once row `row` of `column` (of the key
    /// type) is the last it keeps, ordered as `descending` and `nulls_first`
    /// say. `None` when it would drop nothing: a NULL key under `NULLS LAST`,
    /// a NaN key, or a key type with no filter.
    #[must_use]
    pub fn top_n_bound(
        column: &dyn Array,
        row: usize,
        key_type: &DataType,
        descending: bool,
        nulls_first: bool,
    ) -> Option<Self> {
        let keys = normalize_key_bytes(column, key_type)?;
        let filter = |bound, empty| Self {
            key_type: key_type.clone(),
            range: None,
            values: None,
            member: Member::Range,
            nulls_pass: nulls_first,
            empty,
            keys: 0,
            bound,
        };
        if keys.is_null(row) {
            // Only NULLs rank at or before a NULL under `NULLS FIRST`.
            return nulls_first.then(|| filter(None, true));
        }
        let value = match &keys.norm {
            Norm::Int { values, .. } => KeyValue::Fixed(i128::from(*values.get(row)?)),
            Norm::Decimal(values) => KeyValue::Fixed(*values.get(row)?),
            Norm::Float { values, single } => {
                let v = *values.get(row)?;
                if v.is_nan() {
                    return None;
                }
                KeyValue::Fixed(float_bits(v, *single))
            }
            Norm::Bytes(a) => {
                let mut found = None;
                for_each_bytes(a.as_ref(), |i, b| {
                    if i == row {
                        found = Some(Box::from(b));
                    }
                });
                KeyValue::Bytes(found?)
            }
        };
        Some(filter(Some(TopNBound { value, descending }), false))
    }

    /// Whether this bound drops rows `other`, a bound of the same filter,
    /// keeps. A filter of NULLs only is the tightest there is.
    #[must_use]
    pub fn tighter_than(&self, other: &Self) -> bool {
        if other.empty {
            return false;
        }
        if self.empty {
            return true;
        }
        let float = match self.key_type {
            DataType::Float32 => Some(true),
            DataType::Float64 => Some(false),
            _ => None,
        };
        match (&self.bound, &other.bound) {
            (Some(a), Some(b)) => a.tighter_than(b, float),
            _ => false,
        }
    }
}
