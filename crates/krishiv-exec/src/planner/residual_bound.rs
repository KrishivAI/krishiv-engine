// SPDX-License-Identifier: Apache-2.0
//! Runtime filters from a hash join's comparison residual (design
//! P3-0.3.1 D2, TPC-DS q72's `inv_quantity_on_hand < cs_quantity`).
//!
//! A conjunct `p op b` of the residual, with `p` a probe-side site (a scan
//! column, or a widening cast of one) and `b` a plain build column of the
//! comparison's type, holds for every pair the join returns. So a probe row
//! with `p > max(b)` (for `<`, `<=`) or `p < min(b)` (for `>`, `>=`) over
//! the rows the build stores matches nothing, and its scan may drop it:
//! the build collects that one bound ([`RangeSide`]) through the key
//! filter's range, with no value set or Bloom filter. One bound over all
//! the build's rows is a superset of each key's, so no answer changes, as
//! DuckDB pushes the build's min/max for a comparison join condition
//! (Spark's runtime filters are equality-only). NULL build values are not
//! collected (`x < NULL` matches nothing), and a NULL probe value fails the
//! comparison. Floats take only `Upper` (a NaN probe value is larger than
//! every value in Spark, which the range's compares would drop); strings,
//! binary and booleans none (a collation orders strings otherwise).
//!
//! Planned only where key filters are (joins that drop unmatched probe
//! rows), and only when no residual conjunct or probe key may raise: a probe row the
//! bound drops may still match a key, and Spark would evaluate the other
//! conjuncts on that pair. `krishiv.runtimeFilter.residualBounds`.

use std::sync::Arc;

use arrow_schema::DataType;
use krishiv_io::{KeyFilterOptions, RangeSide};
use krishiv_plan::{AttrId, Builtin, Expr, may_raise};

use super::build::{Planned, Planner, column_of};
use super::pushdown::builtin;
use super::runtime_filter::{JoinFilters, column_type, key_source};
use crate::physical::{JoinFilterPlan, ScanRuntimeFilter};

/// The comparison of `p op b` a bound keeps, with `p` the probe value:
/// `None` for other operators.
fn side_of(op: Builtin, probe_left: bool) -> Option<RangeSide> {
    let upper = match op {
        Builtin::Lt | Builtin::LtEq => true,
        Builtin::Gt | Builtin::GtEq => false,
        _ => return None,
    };
    Some(if upper == probe_left {
        RangeSide::Upper
    } else {
        RangeSide::Lower
    })
}

/// Whether a bound on a build column of type `t` is sound on `side`.
fn bound_type(t: &DataType, side: RangeSide) -> bool {
    use DataType as D;
    match t {
        D::Int8
        | D::Int16
        | D::Int32
        | D::Int64
        | D::Date32
        | D::Decimal64(..)
        | D::Decimal128(..) => true,
        D::Timestamp(..) => krishiv_io::runtime_filter::supported_key_type(t),
        D::Float32 | D::Float64 => side == RangeSide::Upper,
        _ => false,
    }
}

/// One conjunct as a bound: the probe expression, the build attribute and
/// the side.
fn bound_of<'e>(conjunct: &'e Expr, build: &Planned) -> Option<(&'e Expr, AttrId, RangeSide)> {
    let (op, args) = builtin(conjunct)?;
    let [l, r] = args.as_slice() else {
        return None;
    };
    let on_build = |e: &Expr| column_of(e).filter(|a| build.layout.contains(a));
    let (p, b, probe_left) = match (on_build(l), on_build(r)) {
        (None, Some(b)) => (*l, b, true),
        (Some(b), None) => (*r, b, false),
        _ => return None,
    };
    Some((p, b, side_of(op, probe_left)?))
}

impl Planner<'_> {
    /// Adds to `out` the bounds a hash join's `residual` conjuncts publish
    /// for `probe`'s scans (see the module docs), given the probe side's
    /// key expressions `keys`, the build's plan and estimated rows.
    pub(crate) fn residual_bounds(
        &mut self,
        (probe, build): (&Planned, &Planned),
        (keys, residual): (&[Arc<Expr>], &[&Expr]),
        build_rows: f64,
        out: &mut JoinFilters,
    ) {
        let s = self.rf;
        if !s.enabled() || !s.residual_bounds || residual.is_empty() {
            return;
        }
        // A probe key computed above the scan (`CAST(p.s AS BIGINT)`) is
        // evaluated on every row the scan keeps: a row the bound drops
        // would not raise its error.
        let ansi = self.env.ansi;
        if residual.iter().any(|c| may_raise(c, ansi)) || keys.iter().any(|k| may_raise(k, ansi)) {
            return;
        }
        let input_type = |id: AttrId| column_type(probe, id);
        for conjunct in residual {
            let Some((p, b, side)) = bound_of(conjunct, build) else {
                continue;
            };
            let (Some(key_type), Ok(column)) = (column_type(build, b), build.column(b)) else {
                continue;
            };
            if !bound_type(&key_type, side) {
                continue;
            }
            let Some(source) = key_source(p, &input_type) else {
                continue;
            };
            for site in probe.sites.iter().filter(|s| s.attr == source) {
                let id = krishiv_io::FilterId(self.next_filter);
                self.next_filter += 1;
                self.scan_filters
                    .entry(site.scan)
                    .or_default()
                    .push(ScanRuntimeFilter {
                        id,
                        column: site.column.clone(),
                        key_type: key_type.clone(),
                    });
                let (bound, agg) = match side {
                    RangeSide::Lower => (">=", "min"),
                    _ => ("<=", "max"),
                };
                let build_name = build.schema.field(column).name().clone();
                out.explain.push(format!(
                    "RuntimeFilter {id}: {p} {bound} {agg}({build_name}) -> {}.{}",
                    site.scan_name,
                    site.column.0.join("."),
                ));
                out.filters.push(JoinFilterPlan {
                    id,
                    key: 0,
                    options: KeyFilterOptions {
                        max_values: 0,
                        exact_max_values: 0,
                        bloom: false,
                        scan_waits: s.scan_waits(site, &build.node, build_rows),
                        side,
                        ..KeyFilterOptions::default()
                    },
                    column: Some(column),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_side_follows_the_operator_and_where_the_probe_value_is() {
        assert_eq!(side_of(Builtin::Lt, true), Some(RangeSide::Upper));
        assert_eq!(side_of(Builtin::LtEq, true), Some(RangeSide::Upper));
        assert_eq!(side_of(Builtin::Gt, true), Some(RangeSide::Lower));
        assert_eq!(side_of(Builtin::GtEq, true), Some(RangeSide::Lower));
        // `b < p` is `p > b`.
        assert_eq!(side_of(Builtin::Lt, false), Some(RangeSide::Lower));
        assert_eq!(side_of(Builtin::GtEq, false), Some(RangeSide::Upper));
        assert_eq!(side_of(Builtin::Eq, true), None);
    }

    #[test]
    fn bounds_only_on_types_whose_order_the_range_keeps() {
        use DataType as D;
        for t in [D::Int32, D::Int64, D::Date32, D::Decimal128(7, 2)] {
            assert!(bound_type(&t, RangeSide::Upper) && bound_type(&t, RangeSide::Lower));
        }
        let micros = D::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into()));
        assert!(bound_type(&micros, RangeSide::Lower));
        let nanos = D::Timestamp(arrow_schema::TimeUnit::Nanosecond, None);
        assert!(!bound_type(&nanos, RangeSide::Upper));
        assert!(bound_type(&D::Float64, RangeSide::Upper));
        assert!(!bound_type(&D::Float64, RangeSide::Lower));
        for t in [D::Utf8, D::Utf8View, D::Binary, D::Boolean] {
            assert!(!bound_type(&t, RangeSide::Upper));
        }
    }
}
