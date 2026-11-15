// SPDX-License-Identifier: Apache-2.0
//! PySpark functions added to the wrapper table after the first batch, each one `pyspark.sql.functions`
//! signature of Spark 4.2.0. A function the registry implements calls it; one that only Spark's catalog
//! lists (sketches, bitmaps, `xpath`, `like`, `grouping`, ...) gets a wrapper that raises the unsupported-feature
//! error when called, until the registry gains it.

use super::Row;

/// The rows.
pub(super) fn rows() -> Vec<Row> {
    vec![
        Row::new("java_method", "*cols: ColumnOrName"),
        Row::new("reflect", "*cols: ColumnOrName"),
        Row::new("try_reflect", "*cols: ColumnOrName"),
        Row::new(
            "approxCountDistinct",
            "col: ColumnOrName, rsd: Column | float = None",
        )
        .to("approx_count_distinct")
        .deprecated("Deprecated in 2.1, use approx_count_distinct instead."),
        Row::new("negate", "col: ColumnOrName").to("negative"),
        Row::new(
            "power",
            "col1: ColumnOrName | float, col2: ColumnOrName | float",
        ),
        Row::new(
            "timestamp_add",
            "unit: str, quantity: ColumnOrName, ts: ColumnOrName",
        )
        .to("timestampadd"),
        Row::new(
            "timestamp_diff",
            "unit: str, start: ColumnOrName, end: ColumnOrName",
        )
        .to("timestampdiff"),
        Row::new(
            "any_value",
            "col: ColumnOrName, ignoreNulls: bool | Column = None",
        ),
        Row::new("array_agg", "col: ColumnOrName"),
        Row::new("bitmap_and_agg", "col: ColumnOrName"),
        Row::new("bitmap_bit_position", "col: ColumnOrName"),
        Row::new("bitmap_bucket_number", "col: ColumnOrName"),
        Row::new("bitmap_construct_agg", "col: ColumnOrName"),
        Row::new("bitmap_count", "col: ColumnOrName"),
        Row::new("bitmap_or_agg", "col: ColumnOrName"),
        Row::new("chr", "n: ColumnOrName"),
        Row::new("collate", "col: ColumnOrName, collation: str"),
        Row::new(
            "count_min_sketch",
            "col: ColumnOrName, eps: Column | float, confidence: Column | float, seed: Column | int = None",
        ),
        Row::new("every", "col: ColumnOrName"),
        Row::new(
            "from_csv",
            "col: ColumnOrName, schema: Column | str, options: dict[str, str] = None",
        ),
        Row::new(
            "from_xml",
            "col: ColumnOrName, schema: Column | str, options: dict[str, str] = None",
        ),
        Row::new("grouping", "col: ColumnOrName"),
        Row::new("grouping_id", "*cols: ColumnOrName"),
        Row::new("histogram_numeric", "col: ColumnOrName, nBins: Column"),
        Row::new(
            "hll_sketch_agg",
            "col: ColumnOrName, lgConfigK: Column | int = None",
        ),
        Row::new("hll_sketch_estimate", "col: ColumnOrName"),
        Row::new(
            "hll_union",
            "col1: ColumnOrName, col2: ColumnOrName, allowDifferentLgConfigK: bool = None",
        ),
        Row::new(
            "hll_union_agg",
            "col: ColumnOrName, allowDifferentLgConfigK: bool | Column = None",
        ),
        Row::new(
            "ilike",
            "str: ColumnOrName, pattern: ColumnOrName, escapeChar: Column = None",
        ),
        Row::new("is_valid_variant", "v: ColumnOrName"),
        Row::new(
            "kll_merge_agg_bigint",
            "col: ColumnOrName, k: Column | int = None",
        ),
        Row::new(
            "kll_merge_agg_double",
            "col: ColumnOrName, k: Column | int = None",
        ),
        Row::new(
            "kll_merge_agg_float",
            "col: ColumnOrName, k: Column | int = None",
        ),
        Row::new(
            "kll_sketch_agg_bigint",
            "col: ColumnOrName, k: Column | int = None",
        ),
        Row::new(
            "kll_sketch_agg_double",
            "col: ColumnOrName, k: Column | int = None",
        ),
        Row::new(
            "kll_sketch_agg_float",
            "col: ColumnOrName, k: Column | int = None",
        ),
        Row::new("kll_sketch_get_n_bigint", "col: ColumnOrName"),
        Row::new("kll_sketch_get_n_double", "col: ColumnOrName"),
        Row::new("kll_sketch_get_n_float", "col: ColumnOrName"),
        Row::new(
            "kll_sketch_get_quantile_bigint",
            "sketch: ColumnOrName, rank: ColumnOrName",
        ),
        Row::new(
            "kll_sketch_get_quantile_double",
            "sketch: ColumnOrName, rank: ColumnOrName",
        ),
        Row::new(
            "kll_sketch_get_quantile_float",
            "sketch: ColumnOrName, rank: ColumnOrName",
        ),
        Row::new(
            "kll_sketch_get_rank_bigint",
            "sketch: ColumnOrName, quantile: ColumnOrName",
        ),
        Row::new(
            "kll_sketch_get_rank_double",
            "sketch: ColumnOrName, quantile: ColumnOrName",
        ),
        Row::new(
            "kll_sketch_get_rank_float",
            "sketch: ColumnOrName, quantile: ColumnOrName",
        ),
        Row::new(
            "kll_sketch_merge_bigint",
            "left: ColumnOrName, right: ColumnOrName",
        ),
        Row::new(
            "kll_sketch_merge_double",
            "left: ColumnOrName, right: ColumnOrName",
        ),
        Row::new(
            "kll_sketch_merge_float",
            "left: ColumnOrName, right: ColumnOrName",
        ),
        Row::new("kll_sketch_to_string_bigint", "col: ColumnOrName"),
        Row::new("kll_sketch_to_string_double", "col: ColumnOrName"),
        Row::new("kll_sketch_to_string_float", "col: ColumnOrName"),
        Row::new(
            "like",
            "str: ColumnOrName, pattern: ColumnOrName, escapeChar: Column = None",
        ),
        Row::new(
            "max_by",
            "col: ColumnOrName, ord: ColumnOrName, k: int = None",
        ),
        Row::new(
            "min_by",
            "col: ColumnOrName, ord: ColumnOrName, k: int = None",
        ),
        Row::new("rlike", "str: ColumnOrName, regexp: ColumnOrName"),
        Row::new(
            "schema_of_csv",
            "csv: Column | str, options: dict[str, str] = None",
        ),
        Row::new("schema_of_variant_agg", "v: ColumnOrName"),
        Row::new(
            "schema_of_xml",
            "xml: Column | str, options: dict[str, str] = None",
        ),
        Row::new(
            "sentences",
            "string: ColumnOrName, language: ColumnOrName = None, country: ColumnOrName = None",
        ),
        Row::new(
            "session_window",
            "timeColumn: ColumnOrName, gapDuration: Column | str",
        ),
        Row::new("some", "col: ColumnOrName"),
        Row::new(
            "string_agg",
            "col: ColumnOrName, delimiter: Column | str = None",
        ),
        Row::new("theta_difference", "col1: ColumnOrName, col2: ColumnOrName"),
        Row::new(
            "theta_intersection",
            "col1: ColumnOrName, col2: ColumnOrName",
        ),
        Row::new("theta_intersection_agg", "col: ColumnOrName"),
        Row::new(
            "theta_sketch_agg",
            "col: ColumnOrName, lgNomEntries: Column | int = None",
        ),
        Row::new("theta_sketch_estimate", "col: ColumnOrName"),
        Row::new(
            "theta_union",
            "col1: ColumnOrName, col2: ColumnOrName, lgNomEntries: Column | int = None",
        ),
        Row::new(
            "theta_union_agg",
            "col: ColumnOrName, lgNomEntries: Column | int = None",
        ),
        Row::new(
            "to_csv",
            "col: ColumnOrName, options: dict[str, str] = None",
        ),
        Row::new(
            "to_xml",
            "col: ColumnOrName, options: dict[str, str] = None",
        ),
        Row::new(
            "tuple_difference_double",
            "col1: ColumnOrName, col2: ColumnOrName",
        ),
        Row::new(
            "tuple_difference_integer",
            "col1: ColumnOrName, col2: ColumnOrName",
        ),
        Row::new(
            "tuple_difference_theta_double",
            "col1: ColumnOrName, col2: ColumnOrName",
        ),
        Row::new(
            "tuple_difference_theta_integer",
            "col1: ColumnOrName, col2: ColumnOrName",
        ),
        Row::new(
            "tuple_intersection_agg_double",
            "col: ColumnOrName, mode: Column | str = None",
        ),
        Row::new(
            "tuple_intersection_agg_integer",
            "col: ColumnOrName, mode: Column | str = None",
        ),
        Row::new(
            "tuple_intersection_double",
            "col1: ColumnOrName, col2: ColumnOrName, mode: Column | str = None",
        ),
        Row::new(
            "tuple_intersection_integer",
            "col1: ColumnOrName, col2: ColumnOrName, mode: Column | str = None",
        ),
        Row::new(
            "tuple_intersection_theta_double",
            "col1: ColumnOrName, col2: ColumnOrName, mode: Column | str = None",
        ),
        Row::new(
            "tuple_intersection_theta_integer",
            "col1: ColumnOrName, col2: ColumnOrName, mode: Column | str = None",
        ),
        Row::new(
            "tuple_sketch_agg_double",
            "key: ColumnOrName, summary: ColumnOrName, lgNomEntries: Column | int = None, mode: Column | str = None",
        ),
        Row::new(
            "tuple_sketch_agg_integer",
            "key: ColumnOrName, summary: ColumnOrName, lgNomEntries: Column | int = None, mode: Column | str = None",
        ),
        Row::new("tuple_sketch_estimate_double", "col: ColumnOrName"),
        Row::new("tuple_sketch_estimate_integer", "col: ColumnOrName"),
        Row::new(
            "tuple_sketch_summary_double",
            "col: ColumnOrName, mode: Column | str = None",
        ),
        Row::new(
            "tuple_sketch_summary_integer",
            "col: ColumnOrName, mode: Column | str = None",
        ),
        Row::new("tuple_sketch_theta_double", "col: ColumnOrName"),
        Row::new("tuple_sketch_theta_integer", "col: ColumnOrName"),
        Row::new(
            "tuple_union_agg_double",
            "col: ColumnOrName, lgNomEntries: Column | int = None, mode: Column | str = None",
        ),
        Row::new(
            "tuple_union_agg_integer",
            "col: ColumnOrName, lgNomEntries: Column | int = None, mode: Column | str = None",
        ),
        Row::new(
            "tuple_union_double",
            "col1: ColumnOrName, col2: ColumnOrName, lgNomEntries: Column | int = None, mode: Column | str = None",
        ),
        Row::new(
            "tuple_union_integer",
            "col1: ColumnOrName, col2: ColumnOrName, lgNomEntries: Column | int = None, mode: Column | str = None",
        ),
        Row::new(
            "tuple_union_theta_double",
            "col1: ColumnOrName, col2: ColumnOrName, lgNomEntries: Column | int = None, mode: Column | str = None",
        ),
        Row::new(
            "tuple_union_theta_integer",
            "col1: ColumnOrName, col2: ColumnOrName, lgNomEntries: Column | int = None, mode: Column | str = None",
        ),
        Row::new("xpath", "xml: ColumnOrName, path: ColumnOrName"),
        Row::new("xpath_boolean", "xml: ColumnOrName, path: ColumnOrName"),
        Row::new("xpath_double", "xml: ColumnOrName, path: ColumnOrName"),
        Row::new("xpath_float", "xml: ColumnOrName, path: ColumnOrName"),
        Row::new("xpath_int", "xml: ColumnOrName, path: ColumnOrName"),
        Row::new("xpath_long", "xml: ColumnOrName, path: ColumnOrName"),
        Row::new("xpath_number", "xml: ColumnOrName, path: ColumnOrName"),
        Row::new("xpath_short", "xml: ColumnOrName, path: ColumnOrName"),
        Row::new("xpath_string", "xml: ColumnOrName, path: ColumnOrName"),
    ]
}
