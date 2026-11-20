// SPDX-License-Identifier: Apache-2.0
//! The metric-view cases not matched yet (see `metric_views.rs`), by what they need.

/// Cases of both releases.
pub(crate) const GAPS: &[&str] = &[
    // DDL and create checks: DESCRIBE AS JSON, CACHE TABLE, the create-time aggregate and lateral
    // checks.
    "ddl_cache",
    "ddl_describe_as_json",
    "ddl_metric_view_over_metric_view",
    "yaml_dimension_bool",
    "yaml_dimension_is_aggregate",
    "yaml_dimension_number",
    // The read path: EXPLAIN, views over a metric view, the captured eval modes, Spark's
    // analysis-time failures not reproduced yet.
    "read_ansi_cast_in_dimension",
    "read_ansi_created_off_read_on",
    "read_ansi_divide_by_zero_dimension",
    "read_ansi_overflow_reader_conf",
    "read_dimension_names_other_dimension",
    "read_explain_aggregate",
    "read_explain_select_star",
    "read_measure_in_where",
    "read_order_by_measure_not_selected",
    "read_plain_view_aggregate_over_select_star",
    "read_self_join",
];

/// Cases of 4.3.0 alone: the session has no Spark 4.3 profile yet, so the missing-clause error is
/// 4.2's `INTERNAL_ERROR` (design note §1).
pub(crate) const GAPS_4_3: &[&str] = &[
    "parse_missing_both_clauses",
    "parse_missing_language",
    "parse_missing_with_metrics",
];
