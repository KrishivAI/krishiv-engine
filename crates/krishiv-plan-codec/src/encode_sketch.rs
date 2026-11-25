// SPDX-License-Identifier: Apache-2.0
//! The wire form of the HLL, Theta and KLL sketch aggregates: the protobuf function and the folded parameter
//! (`hll_precision`: lgConfigK, allowDifferentLgConfigK as 0 or 1, lgNomEntries, or `k`, 0 for a KLL merge without
//! one and for a Theta intersection).

use std::cell::Cell;

use krishiv_exec::aggregate::{AggregateExpr, AggregateFunction, KllType};

use crate::proto as pb;

fn kll(
    ty: KllType,
    bigint: pb::AggregateFunction,
    float: pb::AggregateFunction,
    double: pb::AggregateFunction,
) -> pb::AggregateFunction {
    match ty {
        KllType::Bigint => bigint,
        KllType::Float => float,
        KllType::Double => double,
    }
}

/// The wire function and folded parameter of a sketch aggregate; any other function is `(Unspecified, 0)` (the
/// caller matched a sketch one).
fn sketch_function(function: &AggregateFunction) -> (pb::AggregateFunction, u32) {
    use pb::AggregateFunction as P;
    match function {
        AggregateFunction::HllSketchAgg { lg_k } => (P::HllSketchAgg, u32::from(*lg_k)),
        AggregateFunction::HllUnionAgg { allow_different } => {
            (P::HllUnionAgg, u32::from(*allow_different))
        }
        AggregateFunction::ThetaSketchAgg { lg_nom } => (P::ThetaSketchAgg, u32::from(*lg_nom)),
        AggregateFunction::ThetaUnionAgg { lg_nom } => (P::ThetaUnionAgg, u32::from(*lg_nom)),
        AggregateFunction::ThetaIntersectionAgg => (P::ThetaIntersectionAgg, 0),
        AggregateFunction::KllSketchAgg { ty, k } => (
            kll(
                *ty,
                P::KllSketchAggBigint,
                P::KllSketchAggFloat,
                P::KllSketchAggDouble,
            ),
            u32::from(*k),
        ),
        AggregateFunction::KllMergeAgg { ty, k } => (
            kll(
                *ty,
                P::KllMergeAggBigint,
                P::KllMergeAggFloat,
                P::KllMergeAggDouble,
            ),
            k.map_or(0, u32::from),
        ),
        _ => (P::Unspecified, 0),
    }
}

/// The wire expression of a sketch aggregate.
pub(crate) fn sketch_expr(a: &AggregateExpr, nested: &Cell<bool>) -> pb::AggregateExpr {
    let (f, k) = sketch_function(&a.function);
    crate::encode::percentile_expr(f, &[], false, false, 0, false, k, a, nested)
}
