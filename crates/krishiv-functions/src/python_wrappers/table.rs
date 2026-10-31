// SPDX-License-Identifier: Apache-2.0
//! The PySpark signature table: Python name, parameters as `pyspark.sql.functions` declares them, and
//! the registry function a call becomes. Taken from `python/pyspark/sql/functions/builtin.py` at Spark
//! 4.0.0, 4.1.0 and 4.2.0 (identical for these names); the registry name and argument order come from
//! `python/pyspark/sql/connect/functions/builtin.py`, which sends SQL function calls as the engine
//! receives them.

use super::Row;

/// The reason the lambda functions have no wrapper: the Python `Column`
/// API cannot build lambdas yet.
const NO_LAMBDA: &str = "takes a lambda; the Python `Column` API cannot build lambdas yet";

/// Registry names that need no wrapper, with the reason.
pub(super) const NOT_EXPORTED: &[(&str, &str)] = &[
    ("&", "operator: `Column.bitwiseAND`"),
    ("|", "operator: `Column.bitwiseOR`"),
    ("^", "operator: `Column.bitwiseXOR`"),
    ("<<", "operator"),
    (">>", "operator"),
    (">>>", "operator"),
    ("if", "no PySpark function"),
    (
        "timestampadd",
        "parsed from `TIMESTAMPADD(unit, n, ts)`; PySpark names it `timestamp_add`",
    ),
    (
        "timestampdiff",
        "parsed from `TIMESTAMPDIFF(unit, a, b)`; PySpark names it `timestamp_diff`",
    ),
    ("mod", "no PySpark function"),
    ("any", "no PySpark function"),
    ("any_value", "no PySpark function"),
    (
        "array_agg",
        "ANSI spelling of `collect_list`; SQL-only, like Spark's naming",
    ),
    ("random", "an alias of `rand`; no PySpark function"),
    ("chr", "no PySpark function; use `char`"),
    ("every", "no PySpark function"),
    ("len", "no PySpark function; use `length`"),
    ("luhn_check", "no PySpark function"),
    ("max_by", "no PySpark function"),
    ("min_by", "no PySpark function"),
    (
        "percentile_cont",
        "ordered-set aggregate: SQL-only, needs WITHIN GROUP (ORDER BY ...)",
    ),
    (
        "percentile_disc",
        "ordered-set aggregate: SQL-only, needs WITHIN GROUP (ORDER BY ...)",
    ),
    ("some", "no PySpark function"),
    ("vector_cosine_similarity", "no PySpark function in 4.2"),
    ("vector_inner_product", "no PySpark function in 4.2"),
    ("vector_l2_distance", "no PySpark function in 4.2"),
    ("vector_norm", "no PySpark function in 4.2"),
    ("vector_normalize", "no PySpark function in 4.2"),
    // The cast-named functions (`float(x)`, `int(x)`, ...): the analyzer
    // rewrites every call to a `CAST`, so there is no call to send.
    (
        "boolean",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "tinyint",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "smallint",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "int",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "bigint",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "float",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "double",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "decimal",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "string",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "binary",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "date",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "timestamp",
        "cast-named function; the analyzer rewrites the call to a cast",
    ),
    (
        "string_agg",
        "an alias of `listagg`; SQL-only, like `array_agg`",
    ),
    ("vector_avg", "no PySpark function; SQL only"),
    ("vector_sum", "no PySpark function; SQL only"),
    (
        "schema_of_csv",
        "PySpark's `options` map parameter has no wrapper support yet; SQL only",
    ),
    (
        "from_csv",
        "PySpark's `options` map parameter has no wrapper support yet; SQL only",
    ),
    (
        "to_csv",
        "PySpark's `options` map parameter has no wrapper support yet; SQL only",
    ),
    (
        "schema_of_xml",
        "PySpark's `options` map parameter has no wrapper support yet; SQL only",
    ),
    (
        "from_xml",
        "PySpark's `options` map parameter has no wrapper support yet; SQL only",
    ),
    (
        "to_xml",
        "PySpark's `options` map parameter has no wrapper support yet; SQL only",
    ),
    ("space", "no PySpark function"),
    (
        "mean",
        "an alias of `avg`; the exported `mean` is a row that sends `avg`",
    ),
    (
        "pow",
        "an alias of `power`; the exported `pow` is a row that sends `power`",
    ),
    ("transform", NO_LAMBDA),
    ("filter", NO_LAMBDA),
    ("exists", NO_LAMBDA),
    ("forall", NO_LAMBDA),
    ("aggregate", NO_LAMBDA),
    ("reduce", NO_LAMBDA),
    ("zip_with", NO_LAMBDA),
    ("map_filter", NO_LAMBDA),
    ("map_zip_with", NO_LAMBDA),
    ("transform_keys", NO_LAMBDA),
    ("transform_values", NO_LAMBDA),
    (
        "array_sort",
        "takes an optional lambda; the Python `Column` API cannot build lambdas yet",
    ),
];

/// Registry names whose wrapper is written by hand in `functions.py`.
pub(super) const HANDWRITTEN: &[(&str, &str)] = &[
    ("count", "`count(\"*\")` is `count(*)`"),
    ("log", "`log(col)` is `ln(col)`; `log(base, col)` is `log`"),
    (
        "convert_timezone",
        "the first parameter is `Optional[Column]` without a default, so `None` omits it",
    ),
    (
        "unix_timestamp",
        "`unix_timestamp()` takes no arguments: the default format must not become the timestamp",
    ),
];

const C: &str = "col: ColumnOrName";
const S: &str = "str: ColumnOrName";

/// Every PySpark function that has a registry counterpart.
pub(super) fn rows() -> Vec<Row> {
    let mut rows = Vec::new();
    for name in [
        "cume_dist",
        "curdate",
        "current_catalog",
        "current_database",
        "current_date",
        "current_schema",
        "current_timestamp",
        "current_timezone",
        "current_user",
        "dense_rank",
        "e",
        "input_file_block_length",
        "input_file_block_start",
        "input_file_name",
        "localtimestamp",
        "monotonically_increasing_id",
        "now",
        "percent_rank",
        "pi",
        "rank",
        "row_number",
        "session_user",
        "spark_partition_id",
        "user",
        "version",
    ] {
        rows.push(Row::new(name, ""));
    }
    for name in [
        "abs",
        "acos",
        "acosh",
        "ascii",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "avg",
        "base64",
        "bin",
        "bit_and",
        "bit_count",
        "bit_or",
        "bit_xor",
        "bool_and",
        "bool_or",
        "count_if",
        "bit_length",
        "cbrt",
        "char",
        "collect_list",
        "collect_set",
        "cos",
        "cosh",
        "cot",
        "crc32",
        "csc",
        "day",
        "dayname",
        "dayofmonth",
        "dayofweek",
        "dayofyear",
        "degrees",
        "exp",
        "expm1",
        "hex",
        "hour",
        "initcap",
        "isnan",
        "isnotnull",
        "isnull",
        "json_array_length",
        "json_object_keys",
        "kurtosis",
        "length",
        "ln",
        "log10",
        "log1p",
        "log2",
        "lower",
        "map_entries",
        "map_from_entries",
        "map_keys",
        "map_values",
        "max",
        "md5",
        "min",
        "minute",
        "month",
        "monthname",
        "negative",
        "nullifzero",
        "octet_length",
        "positive",
        "quarter",
        "quote",
        "radians",
        "reverse",
        "rint",
        "sec",
        "second",
        "sha",
        "sha1",
        "sign",
        "signum",
        "sin",
        "sinh",
        "skewness",
        "soundex",
        "sqrt",
        "std",
        "stddev",
        "stddev_pop",
        "stddev_samp",
        "sum",
        "tan",
        "tanh",
        "unbase64",
        "unhex",
        "unix_date",
        "upper",
        "var_pop",
        "var_samp",
        "variance",
        "weekday",
        "weekofyear",
        "year",
        "zeroifnull",
    ] {
        rows.push(Row::new(name, C));
    }
    for name in ["char_length", "character_length", "lcase", "ucase"] {
        rows.push(Row::new(name, S));
    }
    for name in [
        "is_valid_utf8",
        "make_valid_utf8",
        "try_url_decode",
        "try_validate_utf8",
        "url_decode",
        "url_encode",
        "validate_utf8",
    ] {
        rows.push(Row::new(name, S));
    }
    for name in ["try_avg", "try_sum"] {
        rows.push(Row::new(name, C));
    }
    for name in [
        "regr_avgx",
        "regr_avgy",
        "regr_count",
        "regr_intercept",
        "regr_r2",
        "regr_slope",
        "regr_sxx",
        "regr_sxy",
        "regr_syy",
    ] {
        rows.push(Row::new(name, "y: ColumnOrName, x: ColumnOrName"));
    }
    rows.push(Row::new(
        "listagg",
        "col: ColumnOrName, delimiter: Column | str = None",
    ));
    rows.push(Row::new(
        "mode",
        "col: ColumnOrName, deterministic: bool = False",
    ));
    for name in ["bit_get", "getbit"] {
        rows.push(Row::new(name, "col: ColumnOrName, pos: ColumnOrName"));
    }
    for name in [
        "corr",
        "covar_pop",
        "covar_samp",
        "equal_null",
        "ifnull",
        "nanvl",
        "nullif",
        "nvl",
    ] {
        rows.push(Row::new(name, "col1: ColumnOrName, col2: ColumnOrName"));
    }
    for name in ["to_char", "to_number", "to_varchar", "try_to_number"] {
        rows.push(Row::new(name, "col: ColumnOrName, format: ColumnOrName"));
    }
    for name in [
        "try_add",
        "try_divide",
        "try_mod",
        "try_multiply",
        "try_subtract",
    ] {
        rows.push(Row::new(name, "left: ColumnOrName, right: ColumnOrName"));
    }
    for name in ["regexp", "regexp_count", "regexp_like", "regexp_substr"] {
        rows.push(Row::new(name, "str: ColumnOrName, regexp: ColumnOrName"));
    }
    for name in ["date_diff", "datediff"] {
        rows.push(Row::new(name, "end: ColumnOrName, start: ColumnOrName"));
    }
    for name in ["date_part", "extract"] {
        rows.push(Row::new(name, "field: Column, source: ColumnOrName"));
    }
    for name in ["date_add", "date_sub", "dateadd"] {
        rows.push(Row::new(
            name,
            "start: ColumnOrName, days: ColumnOrName | int",
        ));
    }
    for name in ["atan2", "hypot"] {
        rows.push(Row::new(
            name,
            "col1: ColumnOrName | float, col2: ColumnOrName | float",
        ));
    }
    for name in ["bround", "ceil", "ceiling", "floor", "round"] {
        rows.push(Row::new(
            name,
            "col: ColumnOrName, scale: Column | int = None",
        ));
    }
    for name in ["first", "last"] {
        rows.push(Row::new(
            name,
            "col: ColumnOrName, ignorenulls: bool = False",
        ));
    }
    for name in ["lag", "lead"] {
        rows.push(
            Row::new(
                name,
                "col: ColumnOrName, offset: int = 1, default: Any = None",
            )
            .always("default"),
        );
    }
    for name in ["first_value", "last_value"] {
        rows.push(Row::new(
            name,
            "col: ColumnOrName, ignoreNulls: bool | Column = None",
        ));
    }
    for name in [
        "array",
        "coalesce",
        "hash",
        "map_concat",
        "named_struct",
        "struct",
        "xxhash64",
    ] {
        rows.push(Row::new(name, "*cols: ColumnOrName"));
    }
    for name in ["lpad", "rpad"] {
        rows.push(Row::new(
            name,
            "col: ColumnOrName, len: Column | int, pad: Column | str",
        ));
    }
    for name in ["shiftleft", "shiftright", "shiftrightunsigned"] {
        rows.push(Row::new(name, "col: ColumnOrName, numBits: int"));
    }
    rows.extend([
        Row::new(
            "assert_true",
            "col: ColumnOrName, errMsg: Column | str = None",
        ),
        Row::new("median", C),
        Row::new(
            "percentile",
            "col: ColumnOrName, percentage: Column | float, frequency: Column | int = 1",
        ),
        Row::new(
            "percentile_approx",
            "col: ColumnOrName, percentage: Column | float, accuracy: Column | int = 10000",
        ),
        Row::new(
            "approx_percentile",
            "col: ColumnOrName, percentage: Column | float, accuracy: Column | int = 10000",
        ),
        Row::new(
            "approx_count_distinct",
            "col: ColumnOrName, rsd: Column | float = None",
        ),
        Row::new("raise_error", "errMsg: Column | str"),
        Row::new("create_map", "*cols: ColumnOrName").to("map"),
        Row::new("map_contains_key", "col: ColumnOrName, value: Any"),
        Row::new("map_from_arrays", "col1: ColumnOrName, col2: ColumnOrName"),
        Row::new(
            "str_to_map",
            "text: ColumnOrName, pairDelim: ColumnOrName = None, keyValueDelim: ColumnOrName = None",
        ),
        Row::new("bitwise_not", C).to("~"),
        Row::new("mean", C).to("avg"),
        Row::new("btrim", "str: ColumnOrName, trim: ColumnOrName = None"),
        Row::new(
            "parse_url",
            "url: ColumnOrName, partToExtract: ColumnOrName, key: ColumnOrName = None",
        ),
        Row::new(
            "try_parse_url",
            "url: ColumnOrName, partToExtract: ColumnOrName, key: ColumnOrName = None",
        ),
        Row::new("contains", "left: ColumnOrName, right: ColumnOrName"),
        Row::new("conv", "col: ColumnOrName, fromBase: int, toBase: int"),
        Row::new("endswith", "str: ColumnOrName, suffix: ColumnOrName"),
        Row::new("format_number", "col: ColumnOrName, d: int"),
        Row::new(
            "from_json",
            "col: ColumnOrName, schema: Column | str, options: dict[str, str] = None",
        ),
        Row::new("get_json_object", "col: ColumnOrName, path: str"),
        Row::new("greatest", "*cols: ColumnOrName").at_least(2),
        Row::new("least", "*cols: ColumnOrName").at_least(2),
        Row::new(
            "mask",
            "col: ColumnOrName, upperChar: ColumnOrName = None, lowerChar: ColumnOrName = None, digitChar: ColumnOrName = None, otherChar: ColumnOrName = None",
        ),
        Row::new("ntile", "n: int"),
        Row::new("rand", "seed: int = None"),
        Row::new("randn", "seed: int = None"),
        Row::new(
            "randstr",
            "length: int | Column, seed: int | Column = None",
        ),
        Row::new("typeof", "col: ColumnOrName"),
        Row::new(
            "uniform",
            "min: ColumnOrName | float, max: ColumnOrName | float, seed: int | Column = None",
        ),
        Row::new("uuid", "seed: int | Column = None"),
        Row::new(
            "nth_value",
            "col: ColumnOrName, offset: int, ignoreNulls: bool = False",
        ),
        Row::new(
            "nvl2",
            "col1: ColumnOrName, col2: ColumnOrName, col3: ColumnOrName",
        ),
        Row::new(
            "pmod",
            "dividend: ColumnOrName | float, divisor: ColumnOrName | float",
        ),
        Row::new(
            "pow",
            "col1: ColumnOrName | float, col2: ColumnOrName | float",
        )
        .to("power"),
        Row::new(
            "regexp_extract",
            "str: ColumnOrName, pattern: str, idx: int",
        ),
        Row::new(
            "regexp_extract_all",
            "str: ColumnOrName, regexp: ColumnOrName, idx: int | Column = None",
        ),
        Row::new("array_size", "col: ColumnOrName"),
        Row::new("array_append", "col: ColumnOrName, value: Any"),
        Row::new("array_prepend", "col: ColumnOrName, value: Any"),
        Row::new("array_insert", "arr: ColumnOrName, pos: ColumnOrName | int, value: Any"),
        Row::new("array_compact", "col: ColumnOrName"),
        Row::new("get", "col: ColumnOrName, index: ColumnOrName | int"),
        Row::new("element_at", "col: ColumnOrName, extraction: Any"),
        Row::new("try_element_at", "col: ColumnOrName, extraction: ColumnOrName"),
        Row::new("size", "col: ColumnOrName"),
        Row::new(
            "split",
            "str: ColumnOrName, pattern: str | Column, limit: int | Column = -1",
        ),
        Row::new(
            "regexp_instr",
            "str: ColumnOrName, regexp: ColumnOrName, idx: int | Column = None",
        ),
        Row::new(
            "regexp_replace",
            "string: ColumnOrName, pattern: str | Column, replacement: str | Column",
        ),
        Row::new(
            "replace",
            "src: ColumnOrName, search: ColumnOrName, replace: ColumnOrName = None",
        ),
        Row::new("sha2", "col: ColumnOrName, numBits: int")
            .one_of("numBits", &[0, 224, 256, 384, 512]),
        Row::new("startswith", "str: ColumnOrName, prefix: ColumnOrName"),
        Row::new(
            "schema_of_json",
            "json: Column | str, options: dict[str, str] = None",
        ),
        Row::new(
            "concat_ws",
            "sep: str, *cols: ColumnOrName",
        ),
        Row::new("json_tuple", "col: ColumnOrName, *fields: str"),
        Row::new("decode", "col: ColumnOrName, charset: str"),
        Row::new("elt", "*inputs: ColumnOrName"),
        Row::new("format_string", "format: str, *cols: ColumnOrName"),
        Row::new("printf", "format: ColumnOrName, *cols: ColumnOrName"),
        Row::new("encode", "col: ColumnOrName, charset: str"),
        Row::new("find_in_set", "str: ColumnOrName, str_array: ColumnOrName"),
        Row::new("instr", "str: ColumnOrName, substr: str | Column"),
        Row::new("left", "str: ColumnOrName, len: ColumnOrName"),
        Row::new(
            "levenshtein",
            "left: ColumnOrName, right: ColumnOrName, threshold: int | Column = None",
        ),
        Row::new("locate", "substr: str, str: ColumnOrName, pos: int = 1"),
        Row::new(
            "overlay",
            "src: ColumnOrName, replace: ColumnOrName, pos: ColumnOrName | int, len: ColumnOrName | int = -1",
        ),
        Row::new(
            "position",
            "substr: ColumnOrName, str: ColumnOrName, start: ColumnOrName = None",
        ),
        Row::new("repeat", "col: ColumnOrName, n: ColumnOrName | int"),
        Row::new("right", "str: ColumnOrName, len: ColumnOrName"),
        Row::new(
            "split_part",
            "src: ColumnOrName, delimiter: ColumnOrName, partNum: ColumnOrName",
        ),
        Row::new(
            "substring_index",
            "str: ColumnOrName, delim: str, count: int",
        ),
        Row::new(
            "to_binary",
            "col: ColumnOrName, format: ColumnOrName = None",
        ),
        Row::new(
            "translate",
            "srcCol: ColumnOrName, matching: str, replace: str",
        ),
        Row::new(
            "try_to_binary",
            "col: ColumnOrName, format: ColumnOrName = None",
        ),
        Row::new(
            "substr",
            "str: ColumnOrName, pos: ColumnOrName, len: ColumnOrName = None",
        ),
        Row::new(
            "substring",
            "str: ColumnOrName, pos: ColumnOrName | int, len: ColumnOrName | int",
        ),
        Row::new("last_day", "date: ColumnOrName"),
        Row::new(
            "add_months",
            "start: ColumnOrName, months: ColumnOrName | int",
        ),
        Row::new(
            "make_date",
            "year: ColumnOrName, month: ColumnOrName, day: ColumnOrName",
        ),
        Row::new("timestamp_seconds", "col: ColumnOrName"),
        Row::new("timestamp_millis", "col: ColumnOrName"),
        Row::new("timestamp_micros", "col: ColumnOrName"),
        Row::new("unix_seconds", "col: ColumnOrName"),
        Row::new("unix_millis", "col: ColumnOrName"),
        Row::new("unix_micros", "col: ColumnOrName"),
        Row::new("date_from_unix_date", "days: ColumnOrName"),
        Row::new(
            "months_between",
            "date1: ColumnOrName, date2: ColumnOrName, roundOff: bool = True",
        ),
        Row::new("next_day", "date: ColumnOrName, dayOfWeek: str"),
        Row::new("trunc", "date: ColumnOrName, format: str"),
        Row::new("date_trunc", "format: str, timestamp: ColumnOrName"),
        Row::new(
            "from_utc_timestamp",
            "timestamp: ColumnOrName, tz: Column | str",
        ),
        Row::new("to_json", "col: ColumnOrName, options: dict[str, str] = None"),
        Row::new(
            "to_utc_timestamp",
            "timestamp: ColumnOrName, tz: Column | str",
        ),
        Row::new("date_format", "date: ColumnOrName, format: str"),
        Row::new("to_date", "col: ColumnOrName, format: str = None"),
        Row::new("try_to_date", "col: ColumnOrName, format: str = None"),
        Row::new("to_timestamp", "col: ColumnOrName, format: str = None"),
        Row::new(
            "make_timestamp",
            "years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: ColumnOrName = None",
        ),
        Row::new(
            "make_timestamp_ltz",
            "years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: ColumnOrName = None",
        ),
        Row::new(
            "try_make_timestamp",
            "years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: ColumnOrName = None",
        ),
        Row::new(
            "try_make_timestamp_ltz",
            "years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName, timezone: ColumnOrName = None",
        ),
        Row::new(
            "make_timestamp_ntz",
            "years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName",
        ),
        Row::new(
            "try_make_timestamp_ntz",
            "years: ColumnOrName, months: ColumnOrName, days: ColumnOrName, hours: ColumnOrName, mins: ColumnOrName, secs: ColumnOrName",
        ),
        Row::new(
            "try_to_timestamp",
            "col: ColumnOrName, format: ColumnOrName = None",
        ),
        Row::new(
            "to_timestamp_ltz",
            "timestamp: ColumnOrName, format: ColumnOrName = None",
        ),
        Row::new(
            "to_timestamp_ntz",
            "timestamp: ColumnOrName, format: ColumnOrName = None",
        ),
        Row::new(
            "to_unix_timestamp",
            "timestamp: ColumnOrName, format: ColumnOrName = None",
        ),
        Row::new(
            "from_unixtime",
            "timestamp: ColumnOrName, format: str = \"yyyy-MM-dd HH:mm:ss\"",
        ),
    ]);
    // Spark's `trim(chars, str)`: PySpark's `(col, trim)` reaches the engine reversed.
    for name in ["ltrim", "rtrim", "trim"] {
        rows.push(Row::new(name, "col: ColumnOrName, trim: ColumnOrName = None").reversed());
    }
    rows.extend([
        Row::new("bitwiseNOT", C)
            .to("~")
            .deprecated("Deprecated in 3.2, use bitwise_not instead."),
        Row::new("shiftLeft", "col: ColumnOrName, numBits: int")
            .to("shiftleft")
            .deprecated("Deprecated in 3.2, use shiftleft instead."),
        Row::new("shiftRight", "col: ColumnOrName, numBits: int")
            .to("shiftright")
            .deprecated("Deprecated in 3.2, use shiftright instead."),
        Row::new("shiftRightUnsigned", "col: ColumnOrName, numBits: int")
            .to("shiftrightunsigned")
            .deprecated("Deprecated in 3.2, use shiftrightunsigned instead."),
        Row::new("toDegrees", C)
            .to("degrees")
            .deprecated("Deprecated in 2.1, use degrees instead."),
        Row::new("toRadians", C)
            .to("radians")
            .deprecated("Deprecated in 2.1, use radians instead."),
    ]);
    rows
}
