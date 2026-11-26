// SPDX-License-Identifier: Apache-2.0
//! The wire form of the aggregates that carry folded parameters: the percentile family, `approx_count_distinct`,
//! the sketch aggregates and `count_min_sketch`.

use std::cell::Cell;

use krishiv_exec::aggregate::{AggInput, AggregateExpr};

use crate::encode::{arrow_bytes, indexes};
use crate::proto as pb;

/// A `count_min_sketch` call with its folded shape and seed.
pub(crate) fn count_min_expr(
    depth: u32,
    width: u32,
    seed: i32,
    a: &AggregateExpr,
    nested: &Cell<bool>,
) -> pb::AggregateExpr {
    let function = pb::AggregateFunction::CountMinSketch;
    pb::AggregateExpr {
        count_min_depth: depth,
        count_min_width: width,
        count_min_seed: seed,
        ..percentile_expr(function, &[], false, false, 0, false, 0, a, nested)
    }
}

/// An [`AggregateExpr`] of the percentile family or `approx_count_distinct`,
/// with its folded parameters.
#[expect(
    clippy::too_many_arguments,
    reason = "one call site, every folded parameter"
)]
pub(crate) fn percentile_expr(
    function: pb::AggregateFunction,
    percentages: &[f64],
    array: bool,
    reverse: bool,
    accuracy: i64,
    legacy: bool,
    precision: u32,
    a: &AggregateExpr,
    nested: &Cell<bool>,
) -> pb::AggregateExpr {
    pb::AggregateExpr {
        function: function as i32,
        ignore_nulls: false,
        args: indexes(&a.args),
        input: match a.input {
            AggInput::Raw => pb::AggregateInput::Raw,
            AggInput::State => pb::AggregateInput::State,
        } as i32,
        arg_type: arrow_bytes(&a.arg_type, nested),
        name: a.name.clone(),
        percentages: percentages.to_vec(),
        percentile_array: array,
        percentile_reverse: reverse,
        percentile_accuracy: accuracy,
        percentile_legacy: legacy,
        hll_precision: precision,
        mask: a.mask.map(|m| m as u64),
        ..Default::default()
    }
}
