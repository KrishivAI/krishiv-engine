# SPDX-License-Identifier: Apache-2.0
"""Statistics and sampling built from the engine's own aggregates and functions.

``describe``/``summary``, ``approxQuantile``, ``corr``, ``cov`` and ``freqItems`` run one engine aggregate
and finish in Python (the result is a handful of values, or a local frame of them); they follow Spark's
``StatFunctions`` and ``FrequentItems``. Sampling draws one ``rand(seed)`` per row, Spark's
``XORShiftRandom`` stream, and keeps the rows whose draw is in ``[lower, upper)`` as
``SampleExec`` does, so on one partition the rows are the ones Spark keeps for the same seed.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from krishiv.errors import (
    AnalysisException,
    IllegalArgumentException,
    NumberFormatException,
    PySparkValueError,
    SparkRuntimeException,
    unsupported_feature,
)
from krishiv.sql import functions as F
from krishiv.sql.column import Column

if TYPE_CHECKING:
    from krishiv.sql.dataframe import DataFrame

#: The most rows ``freqItems`` reads for a column: it counts them in the driver, one Python step per row.
FREQ_ITEMS_MAX_ROWS = 10_000_000
_INT_MAX = 2**31 - 1
_LONG_MIN, _LONG_MAX = -(2**63), 2**63 - 1
_NUMERIC = ("tinyint", "smallint", "int", "bigint", "float", "double")
DEFAULT_SUMMARY = ("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
_DEFAULT_DESCRIBE = ("count", "mean", "stddev", "min", "max")
_NAMED = ("count", "count_distinct", "approx_count_distinct", "mean", "stddev", "min", "max")
_SAMPLE_COLUMN = "__krishiv_sample_rand"
_SPARK_TYPE = {
    "string": "StringType",
    "boolean": "BooleanType",
    "binary": "BinaryType",
    "date": "DateType",
    "timestamp": "TimestampType",
    "timestamp_ntz": "TimestampNTZType",
    "int": "IntegerType",
    "bigint": "LongType",
    "smallint": "ShortType",
    "tinyint": "ByteType",
    "double": "DoubleType",
    "float": "FloatType",
}


def _is_numeric(dtype: str) -> bool:
    return dtype in _NUMERIC or dtype.startswith("decimal")


def _spark_type(dtype: str) -> str:
    return _SPARK_TYPE.get(dtype, dtype)


def _quote(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _require(message: str) -> IllegalArgumentException:
    return IllegalArgumentException(f"requirement failed: {message}")


def _java_double(value: float) -> str:
    """A double as Scala's ``toString`` prints it, for messages."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value != 0 and (abs(value) >= 1e7 or abs(value) < 1e-3):
        mantissa, _, exponent = f"{value:.17g}".partition("e")
        return repr(value) if not exponent else f"{float(mantissa)!r}E{int(exponent)}"
    return repr(value)


# ---- describe / summary -----------------------------------------------------------------------------------


def _percentile(statistic: str) -> float | None:
    """The fraction a ``<number>%`` statistic names, ``None`` for any other name."""
    if not statistic.endswith("%"):
        return None
    text = statistic[:-1]
    if "_" in text or not text.strip():
        raise _unrecognized(statistic)
    try:
        return float(text) / 100.0
    except ValueError:
        raise _unrecognized(statistic) from None


def _unrecognized(statistic: str) -> IllegalArgumentException:
    return IllegalArgumentException(
        f"[UNRECOGNIZED_STATISTIC] The statistic '{statistic}' is not recognized. Valid statistics include "
        "`count`, `count_distinct`, `approx_count_distinct`, `mean`, `stddev`, `min`, `max`, and percentile "
        "values. Percentile must be a numeric value followed by '%', within the range 0% to 100%. SQLSTATE: 42704",
        errorClass="UNRECOGNIZED_STATISTIC",
        messageParameters={"stats": f"'{statistic}'"},
        sqlState="42704",
    )


def _check_statistics(statistics: tuple[str, ...]) -> None:
    percentiles = [p for s in statistics if (p := _percentile(s)) is not None]
    if not all(0.0 <= p <= 1.0 for p in percentiles):
        raise _require("Percentiles must be in the range [0, 1]")
    for s in statistics:
        if _percentile(s) is None and s.lower() not in _NAMED:
            raise _unrecognized(s)
    for i, s in enumerate(statistics):
        if s in statistics[:i]:
            raise SparkRuntimeException(
                f"[DUPLICATED_MAP_KEY] Duplicate map key {s} was found, please check the input data.\n"
                'If you want to remove the duplicated keys, you can set "spark.sql.mapKeyDedupPolicy" to '
                '"LAST_WIN" so that the key inserted at last takes precedence. SQLSTATE: 23505',
                errorClass="DUPLICATED_MAP_KEY",
                messageParameters={"key": s, "mapKeyDedupPolicy": '"spark.sql.mapKeyDedupPolicy"'},
                sqlState="23505",
            )


def _statistic_column(statistic: str, name: str, dtype: str) -> Column:
    """The aggregate of ``statistic`` over column ``name`` of type ``dtype``, as a string. As Spark, the moments
    and percentiles of a string column read it as a double that is NULL where it is not a number."""
    column = F.col(_quote(name))
    as_double = column.try_cast("double") if dtype == "string" else column
    lowered = statistic.lower()
    fraction = _percentile(statistic)
    if fraction is not None:
        base = column if dtype in _NUMERIC else column.cast("double")
        if dtype == "string":
            base = as_double
        agg = F.percentile_approx(base, fraction, 10000)
    elif lowered == "count":
        agg = F.count(column)
    elif lowered == "count_distinct":
        agg = F.countDistinct(column)
    elif lowered == "approx_count_distinct":
        agg = F.approx_count_distinct(column, 0.05)
    elif lowered == "mean":
        agg = F.avg(as_double)
    elif lowered == "stddev":
        agg = F.stddev(as_double)
    elif lowered == "min":
        agg = F.min(column)
    else:
        agg = F.max(column)
    return agg.cast("string")


def summary_frame(df: DataFrame, statistics: tuple[str, ...]) -> DataFrame:
    """``df.summary(*statistics)``: a ``summary`` column of the statistic names and one string column for each
    numeric or string column, evaluated now (one engine aggregate) into a local frame."""
    from krishiv.sql.types import StringType, StructField, StructType

    _check_statistics(statistics)
    dtypes = [(n, t) for n, t in df.dtypes if _is_numeric(t) or t == "string"]
    schema = StructType(
        [StructField("summary", StringType(), True)] + [StructField(n, StringType(), True) for n, _ in dtypes]
    )
    rows: list[tuple[Any, ...]]
    if dtypes and statistics:
        width = len(dtypes)
        cells: dict[tuple[int, int], Any] = {}
        # The engine's rewrite of several DISTINCT aggregates does not take `approx_count_distinct` beside
        # them, so `count_distinct` is aggregated on its own.
        for distinct in (False, True):
            batch = [
                (i, j, _statistic_column(s, n, t).alias(f"c{i}_{j}"))
                for i, s in enumerate(statistics)
                for j, (n, t) in enumerate(dtypes)
                if (s.lower() == "count_distinct") == distinct
            ]
            if batch:
                values = df.agg(*[c for _, _, c in batch]).collect()[0]
                cells.update({(i, j): values[k] for k, (i, j, _) in enumerate(batch)})
        rows = [(s, *[cells[(i, j)] for j in range(width)]) for i, s in enumerate(statistics)]
    else:
        rows = [(s, *[None] * len(dtypes)) for s in statistics]
    return df.sparkSession.createDataFrame(rows, schema)


def describe_frame(df: DataFrame, cols: tuple[str, ...]) -> DataFrame:
    """``df.describe(*cols)``: ``summary`` of count, mean, stddev, min and max over the named columns."""
    if cols:
        lowered = [c.lower() for c in cols]
        for c in cols:
            if lowered.count(c.lower()) > 1:
                raise AnalysisException(
                    f"[AMBIGUOUS_REFERENCE] Reference `{c}` is ambiguous, could be: [`{c}`, `{c}`]. SQLSTATE: 42704",
                    errorClass="AMBIGUOUS_REFERENCE",
                    messageParameters={"name": f"`{c}`", "referenceNames": f"[`{c}`, `{c}`]"},
                    sqlState="42704",
                )
        df = df.select(*cols)
    return summary_frame(df, _DEFAULT_DESCRIBE)


# ---- approxQuantile / corr / cov --------------------------------------------------------------------------


def approx_quantile(
    df: DataFrame, columns: list[str], probabilities: list[float], relative_error: float
) -> list[list[float]]:
    """The quantiles of each of ``columns`` (a list of values for each column), as Spark's
    ``StatFunctions.multipleApproxQuantiles``: NULL and NaN are skipped."""
    if not probabilities:
        raise PySparkValueError(
            "[CANNOT_BE_EMPTY] approxQuantile needs at least one probability; `probabilities` is empty. "
            "(Spark fails with a NullPointerException here.)",
            errorClass="CANNOT_BE_EMPTY",
            messageParameters={"item": "probabilities"},
        )
    for name in columns:
        dtype = _dtype_of(df, name)
        if dtype is not None and not _is_numeric(dtype):
            raise _require(
                f"Quantile calculation for column {name} with data type {_spark_type(dtype)} is not supported."
            )
    accuracy = _INT_MAX if relative_error == 0 else min(_INT_MAX, math.ceil(1.0 / relative_error))
    # One scalar aggregate for each column and probability (the engine takes no array of percentages).
    aggregates = []
    for i, name in enumerate(columns):
        value = F.col(_quote(name)).cast("double")
        usable = F.when(~F.isnan(value), value)
        for j, p in enumerate(probabilities):
            aggregates.append(F.percentile_approx(usable, float(p), accuracy).alias(f"q{i}_{j}"))
    row = df.agg(*aggregates).collect()[0]
    width = len(probabilities)
    result = []
    for i in range(len(columns)):
        values = [row[i * width + j] for j in range(width)]
        result.append([] if any(v is None for v in values) else [float(v) for v in values])
    return result


def _dtype_of(df: DataFrame, name: str) -> str | None:
    """The type of column ``name``, matched as the session's resolver matches names (case-insensitively
    unless ``spark.sql.caseSensitive`` is on); ``None`` when it is not a top-level column."""
    case_sensitive = (df.sparkSession.conf.get("spark.sql.caseSensitive", "false") or "false").lower() == "true"
    for column, dtype in df.dtypes:
        if column == name or (not case_sensitive and column.lower() == name.lower()):
            return dtype
    return None


def _require_numeric(df: DataFrame, name: str, what: str) -> None:
    dtype = _dtype_of(df, name)
    if dtype is not None and not _is_numeric(dtype):
        raise _require(f"Currently {what} calculation for columns with dataType {dtype} not supported.")


def _filled(name: str) -> Column:
    """The column as a double, with NULL read as 0.0 (Spark replaces NULL before ``corr`` and ``cov``)."""
    return F.coalesce(F.col(_quote(name)).cast("double"), F.lit(0.0))


def corr(df: DataFrame, col1: str, col2: str) -> float:
    """The Pearson correlation of two numeric columns, NaN where it is undefined; NULL values count as 0.0."""
    _require_numeric(df, col1, "correlation")
    _require_numeric(df, col2, "correlation")
    value = df.agg(F.corr(_filled(col1), _filled(col2)).alias("v")).collect()[0][0]
    return math.nan if value is None else float(value)


def cov(df: DataFrame, col1: str, col2: str) -> float:
    """The sample covariance of two numeric columns, 0.0 where it is undefined; NULL values count as 0.0."""
    _require_numeric(df, col1, "covariance")
    _require_numeric(df, col2, "covariance")
    value = df.agg(F.covar_samp(_filled(col1), _filled(col2)).alias("v")).collect()[0][0]
    return 0.0 if value is None else float(value)


# ---- freqItems ------------------------------------------------------------------------------------------------


class _Nan:
    """A NaN key: NaN never equals itself in Spark's counter map, so each one is its own key."""


def _hashable(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return _Nan()
    if isinstance(value, list | tuple):
        return (type(value).__name__, *[_hashable(v) for v in value])
    if isinstance(value, dict):
        return ("dict", *[(repr(k), _hashable(v)) for k, v in value.items()])
    return value


def _count_item(counts: dict[Any, int], originals: dict[Any, Any], value: Any, size: int) -> None:
    """One row of Spark's ``CollectFrequentItems.add`` (a count of one)."""
    key = _hashable(value)
    if key in counts:
        counts[key] += 1
    elif len(counts) < size:
        counts[key] = 1
        originals[key] = value
    else:
        smallest = min(counts.values()) if counts else 0
        if 1 - smallest >= 0:
            counts[key] = 1
            originals[key] = value
            for gone in [k for k, v in counts.items() if v <= smallest]:
                del counts[gone]
                del originals[gone]
            for k in counts:
                counts[k] -= smallest
        else:
            for k in counts:
                counts[k] -= 1


def frequent_items(df: DataFrame, cols: list[str], support: float) -> DataFrame:
    """``freqItems``: Spark's ``CollectFrequentItems`` counting (Karp, Schenker and Papadimitriou) over each
    column in row order, so on one partition the items are those Spark keeps (the order of the items is not
    Spark's hash order). The column is read into Python; the result is a local one-row frame."""
    from krishiv.sql.types import ArrayType, StructField, StructType

    if not (1e-4 <= support <= 1.0):
        raise _require(f"Support must be in [1e-4, 1], but got {_java_double(support)}.")
    size = int(1 / support)
    fields = []
    result: list[Any] = []
    plain = True
    for name in cols:
        column = df.select(F.col(_quote(name)))
        fields.append(StructField(f"{name}_freqItems", ArrayType(column.schema.fields[0].dataType, True), False))
        counts: dict[Any, int] = {}
        originals: dict[Any, Any] = {}
        for rows_read, (value,) in enumerate(column.toLocalIterator(), start=1):
            if rows_read > FREQ_ITEMS_MAX_ROWS:
                raise unsupported_feature(
                    f"DataFrame.freqItems over more than {FREQ_ITEMS_MAX_ROWS} rows (it counts in the driver, "
                    "one Python step per row)"
                )
            _count_item(counts, originals, value, size)
        result.append(list(originals.values()))
        plain = plain and all(v is None or isinstance(v, bool | int | float | str) for v in originals.values())
    if plain:
        # One row of array literals, cast to the columns' types (the array literal of no items is untyped).
        items = [
            F.array(*[F.lit(v) for v in values]).cast(f.dataType.simpleString()).alias(f.name)
            for values, f in zip(result, fields, strict=True)
        ]
        return df.sparkSession.range(1).select(*items)
    return df.sparkSession.createDataFrame([tuple(result)], StructType(fields))


# ---- sampling ---------------------------------------------------------------------------------------------------


def check_seed(seed: int) -> int:
    """``seed`` as a 64-bit integer; Spark parses it as a Scala ``Long``."""
    if not _LONG_MIN <= seed <= _LONG_MAX:
        raise NumberFormatException(f'For input string: "{seed}"')
    return seed


def sample_range(df: DataFrame, lower: float, upper: float, seed: int) -> DataFrame:
    """The rows whose ``rand(seed)`` draw is in ``[lower, upper)``: Spark's ``SampleExec`` without replacement.
    The draw comes from ``XORShiftRandom(seed + partition)``, one per row in row order."""
    drawn = df.select(F.col("*"), F.rand(check_seed(seed)).alias(_SAMPLE_COLUMN))
    draw = F.col(_SAMPLE_COLUMN)
    keep = (draw >= F.lit(lower)) & (draw < F.lit(upper)) if lower > 0.0 else draw < F.lit(upper)
    if upper - lower <= 0.0:
        keep = F.lit(False)
    return drawn.filter(keep).drop(_SAMPLE_COLUMN)


def sample(df: DataFrame, with_replacement: bool, fraction: float, seed: int) -> DataFrame:
    """``DataFrame.sample`` after its arguments are parsed."""
    eps = 1e-6
    if with_replacement:
        if not fraction >= 0.0:
            raise _require(f"Sampling fraction ({_java_double(fraction)}) must be nonnegative with replacement")
        raise unsupported_feature(
            "DataFrame.sample with replacement (Spark's per-row Poisson draw is not reproduced, "
            "and an approximation would give a different row multiset silently)"
        )
    if not (fraction >= 0.0 - eps and fraction <= 1.0 + eps):
        raise _require(f"Sampling fraction ({_java_double(fraction)}) must be on interval [0, 1] without replacement")
    return sample_range(df, 0.0, fraction, seed)


def stratified_sample(df: DataFrame, strata: Column, fractions: list[tuple[Any, float]], seed: int) -> DataFrame:
    """``sampleBy``: a row is kept when its ``rand(seed)`` draw is below its stratum's fraction (0 for a stratum
    that is not listed); every row consumes one draw whatever its stratum, as Spark's."""
    fraction: Column | None = None
    # Spark looks the stratum up in a typed map: a string column matches only string keys, any other only
    # numbers (SQL's `=` would coerce `'1'` and `1`).
    kind = df.select(strata.alias("__krishiv_stratum")).dtypes[0][1]
    fractions = [(k, f) for k, f in fractions if isinstance(k, str) == (kind == "string")]
    for stratum, wanted in fractions:
        hit = strata == F.lit(stratum)
        fraction = F.when(hit, F.lit(wanted)) if fraction is None else fraction.when(hit, F.lit(wanted))
    chosen = F.lit(0.0) if fraction is None else fraction.otherwise(F.lit(0.0))
    drawn = df.select(F.col("*"), F.rand(check_seed(seed)).alias(_SAMPLE_COLUMN))
    return drawn.filter(F.col(_SAMPLE_COLUMN) < chosen).drop(_SAMPLE_COLUMN)


def random_split(df: DataFrame, weights: list[float], seed: int) -> list[DataFrame]:
    """``randomSplit``: the rows ordered within each partition by every sortable column (so a split does not
    depend on the input order), each split the rows whose one shared draw falls in its slice of the normalised
    weights."""
    total = sum(weights)
    if not total > 0.0:
        shown = ",".join(_java_double(float(w)) for w in weights)
        raise _require(f"Sum of weights must be positive, but got [{shown}]")
    bounds = [0.0]
    for w in weights:
        bounds.append(bounds[-1] + w / total)
    sortable = [n for n, t in df.dtypes if "map<" not in t and not t.startswith("map") and t != "variant"]
    # Spark sorts within each partition (no exchange); a column list of none is cached instead.
    ordered = (
        df._new(df._df.sort_within_partitions([F.col(_quote(n))._c for n in sortable])) if sortable else df.cache()
    )
    return [sample_range(ordered, lo, hi, seed) for lo, hi in pairwise(bounds)]


__all__ = [
    "approx_quantile",
    "corr",
    "cov",
    "describe_frame",
    "frequent_items",
    "random_split",
    "sample",
    "stratified_sample",
    "summary_frame",
]
