// SPDX-License-Identifier: Apache-2.0
//! The wire form of the HLL and Theta sketch aggregates, checked on decode (`hll_sketch_agg`, `hll_union_agg`,
//! `theta_sketch_agg`, `theta_union_agg`, `theta_intersection_agg`).

use krishiv_exec::aggregate::AggregateFunction;

use crate::error::{Result, corrupt};
use crate::proto as pb;

/// An `hll_sketch_agg` (`sketch`: `hll_precision` is its lgConfigK, in `4..=21`) or `hll_union_agg` (`hll_precision`
/// is 0 or 1) function from its wire form.
pub(crate) fn hll_sketch_function(
    a: &pb::AggregateExpr,
    sketch: bool,
) -> Result<AggregateFunction> {
    let bad = || corrupt("an HLL sketch aggregate of bad parameters");
    if !a.percentages.is_empty()
        || a.percentile_array
        || a.percentile_reverse
        || a.percentile_accuracy != 0
        || a.percentile_legacy
    {
        return Err(bad());
    }
    if sketch {
        match u8::try_from(a.hll_precision) {
            Ok(lg_k) if (4..=21).contains(&lg_k) => Ok(AggregateFunction::HllSketchAgg { lg_k }),
            _ => Err(bad()),
        }
    } else {
        match a.hll_precision {
            0 | 1 => Ok(AggregateFunction::HllUnionAgg {
                allow_different: a.hll_precision == 1,
            }),
            _ => Err(bad()),
        }
    }
}

/// Which Theta aggregate a wire function names.
#[derive(Clone, Copy)]
pub(crate) enum ThetaKind {
    /// `theta_sketch_agg`.
    Sketch,
    /// `theta_union_agg`.
    Union,
    /// `theta_intersection_agg`.
    Intersection,
}

/// A Theta aggregate from its wire form: `hll_precision` is the lgNomEntries in `4..=26` of a sketch or a union,
/// and 0 for an intersection.
pub(crate) fn theta_sketch_function(
    a: &pb::AggregateExpr,
    kind: ThetaKind,
) -> Result<AggregateFunction> {
    let bad = || corrupt("a Theta sketch aggregate of bad parameters");
    if !a.percentages.is_empty()
        || a.percentile_array
        || a.percentile_reverse
        || a.percentile_accuracy != 0
        || a.percentile_legacy
    {
        return Err(bad());
    }
    let lg_nom = || match u8::try_from(a.hll_precision) {
        Ok(k) if (4..=26).contains(&k) => Ok(k),
        _ => Err(bad()),
    };
    match kind {
        ThetaKind::Sketch => Ok(AggregateFunction::ThetaSketchAgg { lg_nom: lg_nom()? }),
        ThetaKind::Union => Ok(AggregateFunction::ThetaUnionAgg { lg_nom: lg_nom()? }),
        ThetaKind::Intersection if a.hll_precision == 0 => {
            Ok(AggregateFunction::ThetaIntersectionAgg)
        }
        ThetaKind::Intersection => Err(bad()),
    }
}
