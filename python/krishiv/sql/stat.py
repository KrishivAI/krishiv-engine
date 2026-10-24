# SPDX-License-Identifier: Apache-2.0
"""``DataFrameStatFunctions`` (``DataFrame.stat``), as ``pyspark.sql.DataFrameStatFunctions``.

``crosstab`` and ``sampleBy`` are built from the DataFrame facade, so they stay lazy and run in the
engine. ``approxQuantile``, ``corr``, ``cov`` and ``freqItems`` need engine aggregates (variance,
covariance, correlation, approximate percentiles, a frequent-items sketch) that do not exist yet:
they check their arguments as PySpark does and then fail with the engine's own
``KRISHIV_UNSUPPORTED_FEATURE`` error, as ``DataFrame.sample`` does, instead of computing the
statistic in Python.
"""

from __future__ import annotations

import random
import sys
from functools import reduce
from typing import TYPE_CHECKING, Any

from krishiv.errors import (
    AnalysisException,
    IllegalArgumentException,
    PySparkTypeError,
    PySparkValueError,
    not_expected_type,
    unsupported_feature,
)
from krishiv.sql import functions as F
from krishiv.sql.column import Column

if TYPE_CHECKING:
    from krishiv.sql.dataframe import DataFrame
    from krishiv.sql.session import SparkSession

#: The key and default of the most distinct values ``crosstab`` turns into columns, as Spark's pivot.
PIVOT_MAX_VALUES_KEY = "spark.sql.pivotMaxValues"
PIVOT_MAX_VALUES_DEFAULT = 10000

#: The setting is a 32-bit integer in Spark, and ``limit`` takes one (and the label read asks for one more).
_MAX_VALUES = 2**31 - 2


def _pivot_max_values(session: SparkSession) -> int:
    """``spark.sql.pivotMaxValues`` (default 10000), kept within 0 and what a 32-bit ``limit`` can hold; an
    unset or non-numeric value gives the default."""
    raw = session.conf.get(PIVOT_MAX_VALUES_KEY, str(PIVOT_MAX_VALUES_DEFAULT))
    try:
        return min(max(int(raw.strip()), 0), _MAX_VALUES) if raw is not None else PIVOT_MAX_VALUES_DEFAULT
    except ValueError:
        return PIVOT_MAX_VALUES_DEFAULT


def _check_names(**names: Any) -> None:
    for arg_name, value in names.items():
        if not isinstance(value, str):
            raise not_expected_type(arg_name, "str", value)


class DataFrameStatFunctions:
    """Statistics functions of a :class:`DataFrame` (``pyspark.sql.DataFrameStatFunctions``)."""

    def __init__(self, df: DataFrame) -> None:
        self.df = df

    def approxQuantile(self, col: Any, probabilities: Any, relativeError: Any) -> list[Any]:
        """The approximate quantiles of numeric columns; see :meth:`DataFrame.approxQuantile`.

        .. note:: The engine has no approximate-percentile aggregate yet: after checking its
            arguments this fails with ``KRISHIV_UNSUPPORTED_FEATURE``.
        """
        if not isinstance(col, str | list | tuple):
            raise not_expected_type("col", "list, str or tuple", col)
        for name in [col] if isinstance(col, str) else col:
            if not isinstance(name, str):
                raise not_expected_type("columns", "list[str]", name)
        if not isinstance(probabilities, list | tuple):
            raise not_expected_type("probabilities", "list or tuple", probabilities)
        for p in probabilities:
            if not isinstance(p, float | int) or p < 0 or p > 1:
                raise not_expected_type("probabilities", "list[float, int]", p)
        if not isinstance(relativeError, float | int):
            raise not_expected_type("relativeError", "float or int", relativeError)
        if relativeError < 0:
            raise PySparkValueError(
                errorClass="NEGATIVE_VALUE",
                messageParameters={"arg_name": "relativeError", "arg_value": str(relativeError)},
            )
        raise unsupported_feature("DataFrame.approxQuantile")

    def corr(self, col1: str, col2: str, method: str | None = None) -> float:
        """The Pearson correlation of two columns; see :meth:`DataFrame.corr`.

        .. note:: The engine has no correlation aggregate yet: after checking its arguments this
            fails with ``KRISHIV_UNSUPPORTED_FEATURE``.
        """
        _check_names(col1=col1, col2=col2)
        if method and method != "pearson":
            raise PySparkValueError(
                errorClass="VALUE_NOT_ALLOWED",
                messageParameters={"arg_name": "method", "allowed_values": "['pearson']"},
            )
        raise unsupported_feature("DataFrame.corr")

    def cov(self, col1: str, col2: str) -> float:
        """The sample covariance of two columns; see :meth:`DataFrame.cov`.

        .. note:: The engine has no covariance aggregate yet: after checking its arguments this
            fails with ``KRISHIV_UNSUPPORTED_FEATURE``.
        """
        _check_names(col1=col1, col2=col2)
        raise unsupported_feature("DataFrame.cov")

    def crosstab(self, col1: str, col2: str) -> DataFrame:
        """A pair-wise frequency table of two columns; see :meth:`DataFrame.crosstab`."""
        from krishiv.sql.dataframe import _quoted

        _check_names(col1=col1, col2=col2)
        df = self.df
        key_name = f"{col1}_{col2}"
        label_name = f"{key_name}$label"
        first, second = F.col(col1), F.col(col2)
        key = F.when(first.isNull(), F.lit("null")).otherwise(first.cast("string")).alias(key_name)
        stripped = F.call_function("regexp_replace", second.cast("string"), F.lit("`"), F.lit(""))
        label = F.when(second.isNull(), F.lit("null")).otherwise(stripped).alias(label_name)
        pairs = df.select(key, label)
        key_col, label_col = _quoted(key_name), _quoted(label_name)
        # The column labels are the distinct values of the second column: they are read at this call, as
        # Spark's ``pivot`` reads them, to name the output columns. Counting runs in the engine.
        limit = _pivot_max_values(df.sparkSession)
        distinct = pairs.select(label_col).distinct().limit(limit + 1).collect()
        if len(distinct) > limit:
            raise AnalysisException(
                f"[_LEGACY_ERROR_TEMP_1324] The pivot column {col2} has more than {limit} distinct values, "
                f"this could indicate an error. If this was intended, set {PIVOT_MAX_VALUES_KEY} to at least "
                "the number of distinct values of the pivot column.",
                errorClass="_LEGACY_ERROR_TEMP_1324",
                messageParameters={"pivotColumn": col2, "maxValues": str(limit), "config": PIVOT_MAX_VALUES_KEY},
            )
        labels = sorted(row[0] for row in distinct)
        if not labels:
            return pairs.select(key_col).distinct()
        counts = [F.count(F.when(label_col == F.lit(v), F.lit(1))).alias(v) for v in labels]
        return pairs.groupBy(key_col).agg(*counts)

    def freqItems(self, cols: Any, support: float | None = None) -> DataFrame:
        """The frequent items of columns; see :meth:`DataFrame.freqItems`.

        .. note:: The engine has no frequent-items sketch yet: after checking its arguments this
            fails with ``KRISHIV_UNSUPPORTED_FEATURE``.
        """
        if not isinstance(cols, list | tuple):
            raise not_expected_type("cols", "list or tuple", cols)
        raise unsupported_feature("DataFrame.freqItems")

    def sampleBy(self, col: Column | str, fractions: dict[Any, float], seed: int | None = None) -> DataFrame:
        """A stratified sample without replacement; see :meth:`DataFrame.sampleBy`.

        .. note:: The engine cannot sample yet: unless ``fractions`` is empty, the returned
            :class:`DataFrame` fails its first action with ``KRISHIV_UNSUPPORTED_FEATURE`` (from
            :meth:`DataFrame.sample`).
        """
        if isinstance(col, str):
            strata = F.col(col)
        elif isinstance(col, Column):
            strata = col
        else:
            raise not_expected_type("col", "Column or str", col)
        if not isinstance(fractions, dict):
            raise not_expected_type("fractions", "dict", fractions)
        wanted = []
        for stratum, fraction in fractions.items():
            if not isinstance(stratum, float | int | str):
                raise PySparkTypeError(
                    errorClass="DISALLOWED_TYPE_FOR_CONTAINER",
                    messageParameters={
                        "arg_name": "fractions",
                        "arg_type": type(fractions).__name__,
                        "allowed_types": "float, int, str",
                        "item_type": type(stratum).__name__,
                    },
                )
            wanted.append((stratum, float(fraction)))
        if not all(0.0 <= fraction <= 1.0 for _, fraction in wanted):
            raise IllegalArgumentException(f"Fractions must be in [0, 1], but got {fractions}.")
        sampled_seed = seed if seed is not None else random.randint(0, sys.maxsize)
        if not wanted:
            return self.df.filter(F.lit(False))
        # A stratum is a filter and a sample of its own; the rows of a stratum that is not in
        # ``fractions`` are in none of them, as Spark treats its fraction as zero.
        parts = [
            self.df.filter(strata == F.lit(stratum)).sample(False, fraction, sampled_seed)
            for stratum, fraction in wanted
        ]
        return reduce(lambda left, right: left.union(right), parts)
