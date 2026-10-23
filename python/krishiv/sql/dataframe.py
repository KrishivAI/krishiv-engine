# SPDX-License-Identifier: Apache-2.0
"""``DataFrame``: a lazily evaluated query, as ``pyspark.sql.DataFrame``.

Transformations and analysis build and check plans; actions (``collect``, ``show``, ``toArrow``, ``toPandas``,
``count``, ``__arrow_c_stream__``) run the query in-process with the GIL released."""

from __future__ import annotations

import importlib
from functools import reduce
from typing import TYPE_CHECKING, Any, cast

from krishiv import _native
from krishiv.errors import (
    PySparkAttributeError,
    PySparkTypeError,
    PySparkValueError,
    not_expected_type,
    not_implemented,
)
from krishiv.sql import _convert
from krishiv.sql.column import Column, _named
from krishiv.sql.group import GroupedData
from krishiv.sql.row import Row
from krishiv.sql.types import StructType, _struct_from_columns

if TYPE_CHECKING:
    from krishiv.sql.session import SparkSession

_HOW = {
    "": "inner",
    "inner": "inner",
    "cross": "cross",
    "outer": "full",
    "full": "full",
    "fullouter": "full",
    "full_outer": "full",
    "left": "left",
    "leftouter": "left",
    "left_outer": "left",
    "right": "right",
    "rightouter": "right",
    "right_outer": "right",
    "semi": "semi",
    "leftsemi": "semi",
    "left_semi": "semi",
    "anti": "anti",
    "leftanti": "anti",
    "left_anti": "anti",
}


#: The most rows ``show`` prints, as Spark: ``ByteArrayMethods.MAX_ROUNDED_ARRAY_LENGTH - 1``.
_SHOW_MAX = 2**31 - 17

#: The session key of the notebook preview size, as Spark's ``spark.sql.repl.eagerEval.maxNumRows``.
EAGER_MAX_ROWS_KEY = "spark.sql.repl.eagerEval.maxNumRows"

#: The preview size when the key is unset or not a number, as Spark's default of 20.
EAGER_DEFAULT_MAX_ROWS = 20

ColumnOrName = Column | str
ColumnsOrNames = ColumnOrName | list[ColumnOrName] | tuple[ColumnOrName, ...]


def _flatten(cols: tuple[Any, ...]) -> list[Any]:
    """``f("a", "b")`` and ``f(["a", "b"])`` are the same call."""
    if len(cols) == 1 and isinstance(cols[0], list | tuple):
        return list(cols[0])
    return list(cols)


def _eager_max_rows(session: SparkSession) -> int:
    """The notebook preview size: ``spark.sql.repl.eagerEval.maxNumRows`` (default 20). Unset,
    empty or non-numeric values give the default; negatives give no rows."""
    raw = session.conf.get(EAGER_MAX_ROWS_KEY, str(EAGER_DEFAULT_MAX_ROWS))
    if raw is None:
        return EAGER_DEFAULT_MAX_ROWS
    try:
        rows = int(raw.strip())
    except ValueError:
        return EAGER_DEFAULT_MAX_ROWS
    return min(max(rows, 0), _SHOW_MAX)


class DataFrame:
    """A query over a session's data. Transformations return a new DataFrame."""

    def __init__(self, native: _native.DataFrame, session: SparkSession) -> None:
        self._df = native
        self._session = session

    @property
    def sparkSession(self) -> SparkSession:
        return self._session

    # Analysis (no execution).
    @property
    def schema(self) -> StructType:
        """The result schema, by analysis."""
        return _struct_from_columns(self._df.schema())

    @property
    def columns(self) -> list[str]:
        return [name for name, _, _ in self._df.schema()]

    @property
    def dtypes(self) -> list[tuple[str, str]]:
        return [(name, type_) for name, type_, _ in self._df.schema()]

    def printSchema(self) -> None:
        print("root")
        for name, type_, nullable in self._df.schema():
            print(f" |-- {name}: {type_} (nullable = {str(nullable).lower()})")

    # Column access.
    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            return Column(self._df.col(item))
        if isinstance(item, Column):
            return self.filter(item)
        if isinstance(item, list | tuple):
            return self.select(*item)
        if isinstance(item, int):
            return Column(self._df.col(self.columns[item]))
        raise not_expected_type("item", "int, str, Column, list or tuple", item)

    def __getattr__(self, name: str) -> Column:
        if name.startswith("_") or name not in self.columns:
            raise PySparkAttributeError(f"'DataFrame' object has no attribute '{name}'")
        return Column(self._df.col(name))

    # Transformations.
    def _new(self, native: _native.DataFrame) -> DataFrame:
        return DataFrame(native, self._session)

    def __repr__(self) -> str:
        """``DataFrame[a: int, b: string]``: the columns with their types, by analysis only
        (the query does not run)."""
        fields = ", ".join(f"{name}: {type_}" for name, type_ in self.dtypes)
        return f"DataFrame[{fields}]"

    def _repr_html_(self) -> str | None:
        """The first ``spark.sql.repl.eagerEval.maxNumRows`` rows (default 20) as an HTML table,
        for notebooks: column names and types in the header, HTML-escaped values. Without
        pyarrow there is no preview, so ``None`` (the notebook falls back to :meth:`__repr__`)."""
        try:
            _convert.require_pyarrow("_repr_html_")
        except ImportError:
            return None
        rows = _eager_max_rows(self._session)
        table = self.limit(rows).toArrow()
        cells = _convert.table_cells(table, self._session_zone())
        columns = [(name, type_) for name, type_ in self.dtypes]
        return _convert.repr_html_string(columns, cells)

    def select(self, *cols: ColumnsOrNames) -> DataFrame:
        return self._new(self._df.select([_named(c) for c in _flatten(cols)]))

    def selectExpr(self, *expr: str) -> DataFrame:
        raise not_implemented("selectExpr (SQL expression strings)")

    def filter(self, condition: Column | str) -> DataFrame:
        if isinstance(condition, str):
            raise not_implemented("filter/where with a SQL expression string")
        if not isinstance(condition, Column):
            raise not_expected_type("condition", "Column or str", condition)
        return self._new(self._df.filter(condition._c))

    where = filter

    def withColumn(self, colName: str, col: Column) -> DataFrame:
        if not isinstance(col, Column):
            raise not_expected_type("col", "Column", col)
        return self._new(self._df.with_column(colName, col._c))

    def alias(self, alias: str) -> DataFrame:
        return self._new(self._df.alias(alias))

    def groupBy(self, *cols: ColumnsOrNames) -> GroupedData:
        return GroupedData(self._df.group_by([_named(c) for c in _flatten(cols)]), self)

    groupby = groupBy

    def agg(self, *exprs: Column) -> DataFrame:
        return self.groupBy().agg(*exprs)

    def join(
        self,
        other: DataFrame,
        on: str | list[str] | Column | list[Column] | None = None,
        how: str | None = None,
    ) -> DataFrame:
        if not isinstance(other, DataFrame):
            raise not_expected_type("other", "DataFrame", other)
        kind = _HOW.get((how or "inner").lower())
        if kind is None:
            raise PySparkValueError(
                errorClass="VALUE_NOT_ALLOWED",
                messageParameters={"arg_name": "how", "allowed_values": ", ".join(sorted(_HOW))},
            )
        if on is not None and kind == "cross":
            kind = "inner"  # Spark: a cross join with a condition is an inner join
        if on is None:
            if kind not in ("inner", "cross"):
                raise PySparkValueError("a join without a condition must be inner or cross")
            return self._new(self._df.join(other._df, how="cross"))
        items = list(on) if isinstance(on, list | tuple) else [on]
        if all(isinstance(i, str) for i in items):
            return self._new(self._df.join(other._df, using=cast("list[str]", items), how=kind))
        if all(isinstance(i, Column) for i in items):
            condition: Column = reduce(lambda a, b: a & b, cast("list[Column]", items))
            return self._new(self._df.join(other._df, on=condition._c, how=kind))
        raise PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "arg_name": "on",
                "expected_type": "str, Column or a list of one kind",
                "arg_type": "mixed list",
            },
        )

    def crossJoin(self, other: DataFrame) -> DataFrame:
        return self.join(other, how="cross")

    def orderBy(self, *cols: ColumnsOrNames, ascending: bool | list[bool] = True) -> DataFrame:
        keys = _flatten(cols)
        orders = ascending if isinstance(ascending, list) else [ascending] * len(keys)
        if len(orders) != len(keys):
            raise PySparkValueError("ascending must have one value per sort column")
        ordered = [
            _named(k) if asc else _named(k).sort_order(False, False) for k, asc in zip(keys, orders, strict=True)
        ]
        return self._new(self._df.order_by(ordered))

    sort = orderBy

    def limit(self, num: int) -> DataFrame:
        if not isinstance(num, int):
            raise not_expected_type("num", "int", num)
        return self._new(self._df.limit(num))

    # Actions: each runs the query again (as each PySpark action does), with the GIL released.
    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        """The result as an Arrow C stream capsule (the Arrow PyCapsule interface), so pyarrow, Polars and
        DuckDB read it directly. The query runs now; its batches are computed as the consumer reads them,
        and releasing the stream cancels it."""
        return self._df.execute().__arrow_c_stream__(requested_schema)

    def _session_zone(self) -> Any:
        return _convert.session_zone(self._session.conf.get("spark.sql.session.timeZone", "UTC") or "UTC")

    def toArrow(self) -> Any:
        """The result as a ``pyarrow.Table`` (needs pyarrow)."""
        pa = _convert.require_pyarrow("toArrow")
        stream = self._df.execute()
        try:
            table = pa.table(stream)
        except Exception:
            # The C stream carries only an Arrow error message: raise the engine's own error if that was the cause.
            stream.check_error()
            raise
        stream.check_error()
        return table

    def toPandas(self) -> Any:
        """The result as a ``pandas.DataFrame``, converted from Arrow (needs pyarrow and pandas). Instants are
        wall-clock times in the session time zone without a zone, as Spark's ``toPandas``."""
        table = self.toArrow()
        try:
            importlib.import_module("pandas")  # pandas itself is optional: only checked here
        except ImportError as e:
            raise ImportError("toPandas needs pandas; install it with `pip install krishiv[pandas]`") from e
        return _convert.session_local(table, self._session_zone()).to_pandas()

    def collect(self) -> list[Row]:
        """All result rows (needs pyarrow)."""
        return _convert.rows_of(self.toArrow())

    def take(self, num: int) -> list[Row]:
        return self.limit(num).collect()

    def head(self, n: int | None = None) -> Row | list[Row] | None:
        if n is None:
            rows = self.take(1)
            return rows[0] if rows else None
        return self.take(n)

    def first(self) -> Row | None:
        return cast("Row | None", self.head())

    def count(self) -> int:
        """The number of rows."""
        return self._df.count()

    def show(self, n: int = 20, truncate: bool | int = True, vertical: bool = False) -> None:
        """Prints the first ``n`` rows as a table (needs pyarrow)."""
        if not isinstance(n, int) or isinstance(n, bool):
            raise not_expected_type("n", "int", n)
        if not isinstance(truncate, bool | int):
            raise not_expected_type("truncate", "bool or int", truncate)
        rows = min(max(n, 0), _SHOW_MAX)  # Spark's bounds: MAX_ROUNDED_ARRAY_LENGTH - 1
        cells = _convert.table_cells(self.limit(rows + 1).toArrow(), self._session_zone())
        print(_convert.show_string(self.columns, cells, rows, truncate, vertical), end="")

    def explain(
        self,
        extended: bool | str | None = None,
        mode: str | None = None,
        *,
        analyze: bool = False,
        verbose: bool = False,
    ) -> None:
        """Prints the query's plans, as PySpark's ``explain`` (``extended`` or ``mode``: ``simple``,
        ``extended``, ``codegen``, ``cost``, ``formatted``).

        Krishiv extensions (keyword only): ``analyze=True`` runs the query and prints the physical
        plan with each operator's metrics (``EXPLAIN ANALYZE``); ``verbose=True`` prints the logical
        plan after each optimizer rule that changed it (``EXPLAIN VERBOSE``). Use
        :meth:`explainString` for the text or the JSON document instead of printing.
        """
        print(self.explainString(extended, mode, analyze=analyze, verbose=verbose))

    def explainString(
        self,
        extended: bool | str | None = None,
        mode: str | None = None,
        *,
        analyze: bool = False,
        verbose: bool = False,
        json: bool = False,
    ) -> str:
        """What :meth:`explain` prints; with ``json=True`` (``analyze`` or ``verbose`` only) the
        versioned JSON document (``krishiv.query-profile`` or ``krishiv.explain-verbose``;
        experimental: may change before 1.0)."""
        if extended is not None and mode is not None:
            raise PySparkValueError(
                errorClass="CANNOT_SET_TOGETHER", messageParameters={"arg_list": "extended and mode"}
            )
        for name, flag in (("analyze", analyze), ("verbose", verbose), ("json", json)):
            if not isinstance(flag, bool):
                raise not_expected_type(name, "bool", flag)
        if extended is not None and not isinstance(extended, bool | str):
            raise not_expected_type("extended", "bool or str", extended)
        if mode is not None and not isinstance(mode, str):
            raise not_expected_type("mode", "str", mode)
        chosen = mode if mode is not None else extended if isinstance(extended, str) else None
        if analyze or verbose:
            if analyze and verbose:
                raise PySparkValueError(
                    errorClass="CANNOT_SET_TOGETHER", messageParameters={"arg_list": "analyze and verbose"}
                )
            if extended is not None or mode is not None:
                raise PySparkValueError(
                    errorClass="CANNOT_SET_TOGETHER",
                    messageParameters={"arg_list": "analyze or verbose and extended or mode"},
                )
            return self._df.explain("analyze" if analyze else "verbose", json)
        if json:
            raise PySparkValueError(
                errorClass="CANNOT_SET_TOGETHER", messageParameters={"arg_list": "json and a Spark explain mode"}
            )
        if chosen is None:
            chosen = "extended" if extended is True else "simple"
        chosen = chosen.lower()
        # Only Spark's modes here: Krishiv's run through the keywords above.
        if chosen not in ("simple", "extended", "codegen", "cost", "formatted"):
            raise PySparkValueError(f"Unknown explain mode: {chosen}")
        return self._df.explain(chosen, False)

    def createOrReplaceTempView(self, name: str) -> None:
        """Registers this DataFrame as the temporary view ``name`` (views of reads by path only, so far)."""
        if not isinstance(name, str):
            raise not_expected_type("name", "str", name)
        self._df.create_or_replace_temp_view(name)
