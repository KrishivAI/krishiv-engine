// SPDX-License-Identifier: Apache-2.0
//! The wire form of the HLL sketch aggregates, checked on decode (`hll_sketch_agg`, `hll_union_agg`).

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
