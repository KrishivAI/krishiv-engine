// SPDX-License-Identifier: Apache-2.0
//! The wire form of the aggregate functions that carry parameters (`percentile`, `percentile_approx`,
//! `percentile_disc`, `approx_count_distinct`, `histogram_numeric`), checked on decode; split from `decode.rs`.

use krishiv_exec::aggregate::AggregateFunction;

use crate::error::{Result, corrupt};
use crate::proto as pb;

/// A `percentile` function from its wire form: the percentages must be
/// present, finite and in `[0.0, 1.0]` (the analyzer folds them), and no
/// parameter of another variant may be set.
pub(crate) fn percentile_function(a: &pb::AggregateExpr) -> Result<AggregateFunction> {
    let bad = || corrupt("a `percentile` of bad parameters");
    if a.percentile_accuracy != 0 || a.percentile_legacy || a.hll_precision != 0 {
        return Err(bad());
    }
    if (a.percentages.is_empty() && !a.percentile_array)
        || a.percentages
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err(bad());
    }
    Ok(AggregateFunction::Percentile {
        percentages: a.percentages.clone(),
        array: a.percentile_array,
        reverse: a.percentile_reverse,
    })
}

/// A `percentile_approx` function from its wire form.
pub(crate) fn percentile_approx_function(a: &pb::AggregateExpr) -> Result<AggregateFunction> {
    let bad = || corrupt("a `percentile_approx` of bad parameters");
    if a.percentile_reverse || a.percentile_legacy || a.hll_precision != 0 {
        return Err(bad());
    }
    if a.percentile_accuracy <= 0
        || (a.percentages.is_empty() && !a.percentile_array)
        || a.percentages
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err(bad());
    }
    Ok(AggregateFunction::PercentileApprox {
        percentages: a.percentages.clone(),
        array: a.percentile_array,
        accuracy: a.percentile_accuracy,
    })
}

/// A `percentile_disc` function from its wire form: one percentage.
pub(crate) fn percentile_disc_function(a: &pb::AggregateExpr) -> Result<AggregateFunction> {
    let bad = || corrupt("a `percentile_disc` of bad parameters");
    if a.percentile_accuracy != 0 || a.hll_precision != 0 {
        return Err(bad());
    }
    if (a.percentages.is_empty() && !a.percentile_array)
        || a.percentages
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err(bad());
    }
    Ok(AggregateFunction::PercentileDisc {
        percentages: a.percentages.clone(),
        array: a.percentile_array,
        reverse: a.percentile_reverse,
        legacy: a.percentile_legacy,
    })
}

/// An `approx_count_distinct` function from its wire form.
pub(crate) fn hll_function(a: &pb::AggregateExpr) -> Result<AggregateFunction> {
    let bad = || corrupt("an `approx_count_distinct` of bad parameters");
    if !a.percentages.is_empty()
        || a.percentile_array
        || a.percentile_reverse
        || a.percentile_accuracy != 0
        || a.percentile_legacy
        || !(4..=30).contains(&a.hll_precision)
    {
        return Err(bad());
    }
    Ok(AggregateFunction::ApproxCountDistinct { p: a.hll_precision })
}

/// A `histogram_numeric` function from its wire form.
pub(crate) fn histogram_function(a: &pb::AggregateExpr) -> Result<AggregateFunction> {
    if !a.percentages.is_empty()
        || a.percentile_array
        || a.percentile_reverse
        || a.percentile_accuracy != 0
        || a.percentile_legacy
        || a.hll_precision != 0
        || a.histogram_bins < 2
    {
        return Err(corrupt("a `histogram_numeric` of bad parameters"));
    }
    Ok(AggregateFunction::HistogramNumeric {
        nb: a.histogram_bins,
    })
}
