// SPDX-License-Identifier: Apache-2.0
//! The metric-view cases not matched yet (see `metric_views.rs`), by what they need.

/// Cases of both releases.
pub(crate) const GAPS: &[&str] = &[
    // `DESCRIBE .. AS JSON` of a view.
    "ddl_describe_as_json",
    // Spark's rule order: `WHERE measure(m)` under an aggregate is `MISSING_ATTRIBUTES`, a
    // self-join's qualified name is unresolved.
    "read_measure_in_where",
    "read_self_join",
    // EXPLAIN's physical plan is Krishiv's.
    "read_explain_aggregate",
];
