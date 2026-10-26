// SPDX-License-Identifier: Apache-2.0
//! The PySpark signature table: Python name, parameters as `pyspark.sql.functions` declares them, and
//! the registry function a call becomes. Taken from `python/pyspark/sql/functions/builtin.py` at Spark
//! 4.0.0, 4.1.0 and 4.2.0 (identical for these names); the registry name and argument order come from
//! `python/pyspark/sql/connect/functions/builtin.py`, which sends SQL function calls as the engine
//! receives them.

use super::Row;

/// Registry names that need no wrapper, with the reason.
pub(super) const NOT_EXPORTED: &[(&str, &str)] = &[
    ("&", "operator: `Column.bitwiseAND`"),
    ("|", "operator: `Column.bitwiseOR`"),
    ("^", "operator: `Column.bitwiseXOR`"),
    ("<<", "operator"),
    (">>", "operator"),
    (">>>", "operator"),
    ("if", "no PySpark function"),
    ("mod", "no PySpark function"),
    ("any", "no PySpark function"),
    ("any_value", "no PySpark function"),
    ("bool_and", "no PySpark function"),
    ("bool_or", "no PySpark function"),
    ("count_if", "no PySpark function"),
    ("random", "an alias of `rand`; no PySpark function"),
    ("chr", "no PySpark function; use `char`"),
    ("every", "no PySpark function"),
    ("len", "no PySpark function; use `length`"),
    ("luhn_check", "no PySpark function"),
    ("max_by", "no PySpark function"),
    ("min_by", "no PySpark function"),
    ("some", "no PySpark function"),
    (
        "schema_of_csv",
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
];

/// Registry names whose wrapper is written by hand in `functions.py`.
pub(super) const HANDWRITTEN: &[(&str, &str)] = &[
    ("count", "`count(\"*\")` is `count(*)`"),
    ("log", "`log(col)` is `ln(col)`; `log(base, col)` is `log`"),
    (
        "convert_timezone",
        "the first parameter is `Optional[Column]` without a default, so `None` omits it",
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
        "bit_count",
        "bit_length",
        "cbrt",
        "char",
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
        "kurtosis",
        "length",
        "ln",
        "log10",
        "log1p",
        "log2",
        "lower",
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
        rows.push(Row::new(
            name,
            "col: ColumnOrName, offset: int = 1, default: Any = None",
        ));
    }
    for name in ["first_value", "last_value"] {
        rows.push(Row::new(
            name,
            "col: ColumnOrName, ignoreNulls: bool | Column = None",
        ));
    }
    for name in ["coalesce", "hash", "xxhash64"] {
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
        Row::new("raise_error", "errMsg: Column | str"),
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
            "concat_ws",
            "sep: str, *cols: ColumnOrName",
        ),
        Row::new("decode", "col: ColumnOrName, charset: str"),
        Row::new("elt", "*inputs: ColumnOrName"),
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
        Row::new(
            "to_utc_timestamp",
            "timestamp: ColumnOrName, tz: Column | str",
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
