// SPDX-License-Identifier: Apache-2.0
//! The wire form of the HLL, Theta and KLL sketch aggregates: the protobuf function and the folded parameter
//! (`hll_precision`: lgConfigK, allowDifferentLgConfigK as 0 or 1, lgNomEntries, or `k`, 0 for a KLL merge without
//! one and for a Theta intersection).

use std::cell::Cell;

use krishiv_exec::aggregate::TupleMode as Mode;
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

/// The mode of a Tuple aggregate on the wire.
pub(crate) fn mode_code(m: Mode) -> u32 {
    match m {
        Mode::Sum => 0,
        Mode::Min => 1,
        Mode::Max => 2,
        Mode::AlwaysOne => 3,
    }
}

/// Whether a sketch aggregate's folded parameter is out of range (the encoder refuses it).
pub(crate) fn bad_parameter(function: &AggregateFunction) -> bool {
    match function {
        AggregateFunction::HllSketchAgg { lg_k } => !(4..=21).contains(lg_k),
        AggregateFunction::KllSketchAgg { k, .. }
        | AggregateFunction::KllMergeAgg { k: Some(k), .. } => !(8..=65_535).contains(k),
        AggregateFunction::ThetaSketchAgg { lg_nom }
        | AggregateFunction::ThetaUnionAgg { lg_nom }
        | AggregateFunction::TupleSketchAgg { lg_nom, .. }
        | AggregateFunction::TupleUnionAgg { lg_nom, .. } => !(4..=26).contains(lg_nom),
        _ => false,
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
        AggregateFunction::TupleSketchAgg {
            integer,
            lg_nom,
            mode,
        } => (
            if *integer {
                P::TupleSketchAggInteger
            } else {
                P::TupleSketchAggDouble
            },
            u32::from(*lg_nom) | (mode_code(*mode) << 8),
        ),
        AggregateFunction::TupleUnionAgg {
            integer,
            lg_nom,
            mode,
        } => (
            if *integer {
                P::TupleUnionAggInteger
            } else {
                P::TupleUnionAggDouble
            },
            u32::from(*lg_nom) | (mode_code(*mode) << 8),
        ),
        AggregateFunction::TupleIntersectionAgg { integer, mode } => (
            if *integer {
                P::TupleIntersectionAggInteger
            } else {
                P::TupleIntersectionAggDouble
            },
            mode_code(*mode) << 8,
        ),
        _ => (P::Unspecified, 0),
    }
}

/// The wire expression of a sketch aggregate.
pub(crate) fn sketch_expr(a: &AggregateExpr, nested: &Cell<bool>) -> pb::AggregateExpr {
    let (f, k) = sketch_function(&a.function);
    crate::encode::percentile_expr(f, &[], false, false, 0, false, k, a, nested)
}
