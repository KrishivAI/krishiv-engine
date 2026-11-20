// SPDX-License-Identifier: Apache-2.0
//! The metric-view cases not matched yet (see `metric_views.rs`), by what they need.

/// Cases of both releases.
pub(crate) const GAPS: &[&str] = &[
    // `DESCRIBE .. AS JSON` of a view.
    "ddl_describe_as_json",
    // A command's `TABLE_OR_VIEW_NOT_FOUND` names `default` in its search path, not the current
    // database (every command, not only metric views).
    "yaml_dimension_bool",
    "yaml_dimension_number",
    // The eval modes captured at create: parser-built operators and casts take the creator's ANSI
    // mode, functions the reader's (design note, golden results 1).
    "read_ansi_cast_in_dimension",
    "read_ansi_created_off_read_on",
    "read_ansi_divide_by_zero_dimension",
    "read_ansi_overflow_reader_conf",
    // Spark's rule order: `ORDER BY measure(m)` of a measure not selected works, `WHERE measure(m)`
    // under an aggregate is `MISSING_ATTRIBUTES`, a self-join's qualified name is unresolved.
    "read_measure_in_where",
    "read_order_by_measure_not_selected",
    "read_self_join",
    // EXPLAIN's physical plan is Krishiv's.
    "read_explain_aggregate",
];

/// Cases of 4.3.0 alone: the session has no Spark 4.3 profile yet, so the missing-clause error is
/// 4.2's `INTERNAL_ERROR` (design note §1).
pub(crate) const GAPS_4_3: &[&str] = &[
    "parse_missing_both_clauses",
    "parse_missing_language",
    "parse_missing_with_metrics",
];
